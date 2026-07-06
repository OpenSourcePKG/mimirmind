// llama-parity-dump — minimal llama.cpp client that runs prompt prefill
// once and dumps the per-block hidden state (tensor "l_out-N") to disk.
//
// Companion to mimirmind's MIMIRMIND_PARITY_DUMP env hook. The two
// programs produce binary files with the same layout so `parity-diff`
// can spot the first block where they diverge.
//
// Output layout per block (little-endian):
//   u32  block_idx
//   u32  T          // number of tokens
//   u32  d_model
//   f32  data[T * d_model]   // token-major (row 0 = first token)
//
// Invocation:
//   llama-parity-dump --model PATH --prompt "Hello, world!"
//                     --dump-dir /tmp/dumps/llama

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

namespace {

struct ParityCtx {
    std::string dumpDir;
    // Tensor name → stage tag for the filename (e.g. "Qcur_pos" → "Qcur_pos").
    // The trailing layer index is captured by the regex.
    std::regex  pattern;
    int         dumped{0};
};

bool parity_cb(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* ctx = static_cast<ParityCtx*>(user_data);
    if (t == nullptr || t->name[0] == 0) {
        return false;
    }
    std::cmatch m;
    const bool match = std::regex_match(t->name, m, ctx->pattern);

    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }

    const std::string stage    = m[1].str();   // "attn_norm" / "Qcur_pos" / ...
    const int         blockIdx = std::atoi(m[2].str().c_str());

    // ggml: ne[0] is fastest-varying. The "per-token width" is the product
    // of all dims except the last (which is n_tokens for these tensors).
    // Tensors here are either 2D [d_model, T] or 3D [head_dim, n_head, T].
    std::uint32_t T = 1, d = 1;
    if (ggml_n_dims(t) == 2) {
        d = static_cast<std::uint32_t>(t->ne[0]);
        T = static_cast<std::uint32_t>(t->ne[1]);
    } else if (ggml_n_dims(t) == 3) {
        d = static_cast<std::uint32_t>(t->ne[0] * t->ne[1]);
        T = static_cast<std::uint32_t>(t->ne[2]);
    } else {
        d = static_cast<std::uint32_t>(ggml_nelements(t));
        T = 1;
    }

    const std::size_t nBytes = ggml_nbytes(t);
    std::vector<std::uint8_t> buf(nBytes);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(buf.data(), t->data, nBytes);
    } else {
        ggml_backend_tensor_get(t, buf.data(), 0, nBytes);
    }

    const std::string fname =
        ctx->dumpDir + "/blk" + std::to_string(blockIdx) +
        "-" + stage + ".bin";
    std::ofstream f(fname, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "parity-dump: cannot open %s for writing\n",
                     fname.c_str());
        return true;
    }
    const std::uint32_t header[3] = {
        static_cast<std::uint32_t>(blockIdx), T, d
    };
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    f.write(reinterpret_cast<const char*>(buf.data()),
            static_cast<std::streamsize>(nBytes));
    ++ctx->dumped;

    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string dumpDir;
    std::string explicitTokensCsv;
    // Strip --dump-dir DIR and --tokens CSV out of argv so
    // common_params_parse doesn't see them.
    std::vector<char*> filteredArgv;
    filteredArgv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) {
            dumpDir = argv[++i];
        } else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            explicitTokensCsv = argv[++i];
        } else {
            filteredArgv.push_back(argv[i]);
        }
    }
    if (dumpDir.empty()) {
        std::fprintf(stderr,
                     "usage: %s --model PATH --dump-dir DIR "
                     "(--prompt TEXT | --tokens id1,id2,...)\n",
                     argv[0]);
        return 1;
    }
    std::filesystem::create_directories(dumpDir);

    common_params params;
    common_init();

    if (!common_params_parse(static_cast<int>(filteredArgv.size()),
                             filteredArgv.data(),
                             params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Stage tensors to capture (alternation in regex). One per execution
    // landmark inside a Gemma 4 block. Group 1 = stage name, group 2 = il.
    ParityCtx ctx{
        /*.dumpDir =*/ dumpDir,
        /*.pattern =*/ std::regex(
            R"(^(inp_scaled|attn_norm|Qcur_pos|Kcur_pos|Vcur_normed|attn_post_norm|attn_out|ffn_mlp|ffn_moe|ffn_moe_combined|ffn_post_norm|out_scaled|l_out)(?:-(-?\d+))?$)"),
        /*.dumped  =*/ 0,
    };

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = parity_cb;
    params.cb_eval_user_data = &ctx;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto* model = llama_init->model();
    auto* lctx  = llama_init->context();
    if (model == nullptr || lctx == nullptr) {
        std::fprintf(stderr, "parity-dump: failed to init model\n");
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const bool add_bos       = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens;
    if (!explicitTokensCsv.empty()) {
        // Feed the exact prefill sequence given by the caller. Bypasses
        // the tokenizer + BOS injection so the token IDs match whatever
        // mimirmind logged from its chat-template renderer.
        std::size_t pos = 0;
        while (pos < explicitTokensCsv.size()) {
            std::size_t comma = explicitTokensCsv.find(',', pos);
            if (comma == std::string::npos) comma = explicitTokensCsv.size();
            const std::string one = explicitTokensCsv.substr(pos, comma - pos);
            if (!one.empty()) {
                tokens.push_back(static_cast<llama_token>(std::atoi(one.c_str())));
            }
            pos = comma + 1;
        }
    } else {
        tokens = common_tokenize(lctx, params.prompt, add_bos, true);
    }
    if (tokens.empty()) {
        std::fprintf(stderr, "parity-dump: empty prompt / tokens\n");
        return 1;
    }

    std::printf("parity-dump: prompt=%zu tokens\n", tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        std::printf("  [%zu] = %d\n", i, tokens[i]);
    }

    if (llama_decode(lctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        std::fprintf(stderr, "parity-dump: llama_decode failed\n");
        return 1;
    }

    std::printf("parity-dump: wrote %d block(s) to %s\n",
                ctx.dumped, dumpDir.c_str());

    llama_backend_free();
    return 0;
}