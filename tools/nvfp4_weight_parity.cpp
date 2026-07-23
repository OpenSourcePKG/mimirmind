// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// NVFP4 weight-parity probe. Dequantises selected weight tensors from the
// NVFP4 (ModelOpt safetensors) checkpoint on the host and compares their
// statistics + leading elements against the GGUF checkpoint of the SAME model
// (a different quantisation of the same trained weights). Both sides store a
// linear weight [out, in] with the in-features contiguous, so element k maps
// to the same (out, in) position on both sides — the leading elements must
// agree within quantisation error. A gross mismatch localises the wrong
// weight class (scale / transpose / mis-map) driving the incoherent output.
//
// Host-only: NvFp4Reference + dequantToF32 are pure CPU. Usage:
//   nvfp4_weight_parity <nvfp4_dir> <gguf_file>

#include "compute/Dequant.hpp"
#include "core/gguf/GgufReader.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/modelopt/NvFp4Reference.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsHeader.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace st = mimirmind::core::safetensors;
namespace mo = mimirmind::core::modelopt;
namespace gg = mimirmind::core::gguf;

namespace {

float bf16ToF32(std::uint16_t h) {
    std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct Stats {
    double mn = 1e30, mx = -1e30, sum = 0, sumsq = 0, sumabs = 0;
    std::uint64_t n = 0;
    void add(float v) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v; sumsq += double(v) * v; sumabs += std::fabs(v); ++n;
    }
    double mean() const { return n ? sum / double(n) : 0; }
    double std_() const { double m = mean(); return n ? std::sqrt(sumsq / double(n) - m * m) : 0; }
    double absmean() const { return n ? sumabs / double(n) : 0; }
};

Stats statsOf(const std::vector<float>& v) {
    Stats s;
    for (float x : v) s.add(x);
    return s;
}

void printHead(const char* tag, const std::vector<float>& v, std::uint64_t k = 8) {
    std::printf("    %-6s [", tag);
    for (std::uint64_t i = 0; i < k && i < v.size(); ++i)
        std::printf("% .5f ", v[i]);
    std::printf("]\n");
}

// Dequantise one NVFP4/FP8/passthrough weight tensor to F32, keyed on its
// safetensors dtype. Returns the full tensor.
std::vector<float> dequantNvfp4Side(const st::SafetensorsModel& m, const std::string& base) {
    const std::string wname = base + ".weight";
    const st::SafetensorsTensor* w = m.find(wname);
    if (!w) { std::printf("    [NVFP4] missing %s\n", wname.c_str()); return {}; }
    const auto wb = m.tensorBytes(wname);

    if (w->dtype == st::SafetensorsDtype::U8) {
        // NVFP4: shape [out, in/2 packed]; scales row-major [out, in/16] F8.
        const std::uint64_t rows = w->shape[0];
        const std::uint64_t in   = w->shape[1] * 2;
        const auto sb = m.tensorBytes(base + ".weight_scale");
        const auto gb = m.tensorBytes(base + ".weight_scale_2");
        float global; std::memcpy(&global, gb.data(), 4);
        std::vector<float> out(static_cast<std::size_t>(rows * in));
        mo::dequantNvfp4(wb.data(), sb.data(), global, rows, in, out.data());
        return out;
    }
    if (w->dtype == st::SafetensorsDtype::F8_E4M3) {
        const auto ws = m.tensorBytes(base + ".weight_scale");
        float scale; std::memcpy(&scale, ws.data(), 4);
        std::vector<float> out(static_cast<std::size_t>(w->nelements));
        mo::dequantFp8(wb.data(), scale, w->nelements, out.data());
        return out;
    }
    if (w->dtype == st::SafetensorsDtype::BF16) {
        std::vector<float> out(static_cast<std::size_t>(w->nelements));
        const auto* h = reinterpret_cast<const std::uint16_t*>(wb.data());
        for (std::uint64_t i = 0; i < w->nelements; ++i) out[i] = bf16ToF32(h[i]);
        return out;
    }
    if (w->dtype == st::SafetensorsDtype::F32) {
        std::vector<float> out(static_cast<std::size_t>(w->nelements));
        std::memcpy(out.data(), wb.data(), static_cast<std::size_t>(w->nelements) * 4);
        return out;
    }
    std::printf("    [NVFP4] unhandled dtype for %s\n", wname.c_str());
    return {};
}

class GgufMmap {
public:
    explicit GgufMmap(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { std::perror("open gguf"); return; }
        struct stat sb {};
        fstat(fd, &sb);
        _size = static_cast<std::size_t>(sb.st_size);
        _base = static_cast<const std::uint8_t*>(
            ::mmap(nullptr, _size, PROT_READ, MAP_PRIVATE, fd, 0));
        ::close(fd);
        if (_base == MAP_FAILED) { std::perror("mmap gguf"); _base = nullptr; }
    }
    ~GgufMmap() { if (_base) ::munmap(const_cast<std::uint8_t*>(_base), _size); }
    const std::uint8_t* base() const { return _base; }
private:
    const std::uint8_t* _base = nullptr;
    std::size_t _size = 0;
};

std::vector<float> dequantGgufSide(const gg::GgufReader& r, const GgufMmap& mm,
                                   const std::string& name, std::uint64_t limitElems) {
    const gg::GgufTensor* t = r.findTensor(name);
    if (!t) { std::printf("    [GGUF ] missing %s\n", name.c_str()); return {}; }
    const std::uint8_t* payload = mm.base() + r.tensorDataOffset() + t->fileOffset;
    const std::uint64_t n = limitElems && limitElems < t->nelements ? limitElems : t->nelements;
    // dequantToF32 works per whole tensor; dequant the full tensor then trim.
    std::vector<float> full(static_cast<std::size_t>(t->nelements));
    mimirmind::compute::dequantToF32(t->type, payload, t->nelements, full.data());
    full.resize(static_cast<std::size_t>(n));
    return full;
}

void compare(const char* label, const std::vector<float>& a, const std::vector<float>& b) {
    std::printf("  %s\n", label);
    if (a.empty() || b.empty()) { std::printf("    (one side empty — skip)\n"); return; }
    Stats sa = statsOf(a), sb = statsOf(b);
    std::printf("    NVFP4  n=%-9llu min=% .4f max=% .4f mean=% .5f std=%.5f absmean=%.5f\n",
                (unsigned long long)sa.n, sa.mn, sa.mx, sa.mean(), sa.std_(), sa.absmean());
    std::printf("    GGUF   n=%-9llu min=% .4f max=% .4f mean=% .5f std=%.5f absmean=%.5f\n",
                (unsigned long long)sb.n, sb.mn, sb.mx, sb.mean(), sb.std_(), sb.absmean());
    printHead("nvfp4", a);
    printHead("gguf", b);
    double ratio = sb.absmean() > 1e-12 ? sa.absmean() / sb.absmean() : 0.0;
    std::printf("    absmean ratio NVFP4/GGUF = %.4f  %s\n", ratio,
                (ratio > 0.5 && ratio < 2.0) ? "(ok)" : "<<< SUSPECT");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s <nvfp4_dir> <gguf_file>\n", argv[0]); return 2; }
    const std::string nvfp4Dir = argv[1];
    const std::string ggufPath = argv[2];

    st::SafetensorsModel sm;
    sm.open(nvfp4Dir);
    std::printf("NVFP4: %zu shards, %zu tensors\n", sm.shardCount(), sm.tensorCount());

    gg::GgufReader gr;
    gr.open(ggufPath);
    GgufMmap mm(ggufPath);
    std::printf("GGUF : %zu tensors, dataOffset=%zu\n\n", gr.tensorCount(), gr.tensorDataOffset());

    struct Probe { const char* label; const char* hfBase; const char* gguf; std::uint64_t limit; };
    const Probe probes[] = {
        {"L0 shared_expert.gate_proj (NVFP4)", "model.language_model.layers.0.mlp.shared_expert.gate_proj", "blk.0.ffn_gate_shexp.weight", 4096},
        {"L0 shared_expert.down_proj (NVFP4)", "model.language_model.layers.0.mlp.shared_expert.down_proj", "blk.0.ffn_down_shexp.weight", 4096},
        {"L0 expert0.gate_proj (NVFP4)",       "model.language_model.layers.0.mlp.experts.0.gate_proj",     "blk.0.ffn_gate_exps.weight",  4096},
        {"L0 in_proj_qkv (FP8)",               "model.language_model.layers.0.linear_attn.in_proj_qkv",     "blk.0.attn_qkv.weight",       4096},
        {"L0 out_proj (FP8)",                  "model.language_model.layers.0.linear_attn.out_proj",        "blk.0.ssm_out.weight",        4096},
        {"L0 attn_norm (passthrough)",         "model.language_model.layers.0.input_layernorm",             "blk.0.attn_norm.weight",      4096},
        {"L0 post_attn_norm",                  "model.language_model.layers.0.post_attention_layernorm",    "blk.0.post_attention_norm.weight", 4096},
        {"L0 ssm_norm",                        "model.language_model.layers.0.linear_attn.norm",            "blk.0.ssm_norm.weight",       4096},
        {"L3 attn_q_norm",                     "model.language_model.layers.3.self_attn.q_norm",            "blk.3.attn_q_norm.weight",    4096},
        {"output_norm",                        "model.language_model.norm",                                 "output_norm.weight",          4096},
        {"L0 ssm_conv1d (passthrough)",        "model.language_model.layers.0.linear_attn.conv1d",          "blk.0.ssm_conv1d.weight",     4096},
        {"L0 ssm_alpha <- in_proj_a",          "model.language_model.layers.0.linear_attn.in_proj_a",       "blk.0.ssm_alpha.weight",      4096},
        {"L0 ssm_beta  <- in_proj_b",          "model.language_model.layers.0.linear_attn.in_proj_b",       "blk.0.ssm_beta.weight",       4096},
        {"L0 attn_gate <- in_proj_z (FP8)",    "model.language_model.layers.0.linear_attn.in_proj_z",       "blk.0.attn_gate.weight",      4096},
        {"L0 router ffn_gate_inp",             "model.language_model.layers.0.mlp.gate",                    "blk.0.ffn_gate_inp.weight",   4096},
        {"L3 attn_k",                          "model.language_model.layers.3.self_attn.k_proj",            "blk.3.attn_k.weight",         4096},
        {"L3 attn_v",                          "model.language_model.layers.3.self_attn.v_proj",            "blk.3.attn_v.weight",         4096},
        {"L3 q_proj (full-attn)",              "model.language_model.layers.3.self_attn.q_proj",            "blk.3.attn_q.weight",         4096},
        {"embed_tokens (passthrough)",         "model.language_model.embed_tokens",                         "token_embd.weight",           4096},
        {"lm_head",                            "lm_head",                                                   "output.weight",               4096},
    };

    for (const auto& p : probes) {
        std::vector<float> a = dequantNvfp4Side(sm, p.hfBase);
        if (a.size() > p.limit) a.resize(static_cast<std::size_t>(p.limit));
        std::vector<float> b = dequantGgufSide(gr, mm, p.gguf, p.limit);
        compare(p.label, a, b);
        std::printf("\n");
    }

    // Expert-stacking check: GGUF ffn_gate_exps is [in, out, n_expert] with
    // expert E contiguous at offset E*perExpert. Compare a few experts' NVFP4
    // gate_proj against the matching slice of the stacked GGUF tensor. A wrong
    // stacking offset/order misplaces experts 1..255 -> MoE garbage.
    {
        const gg::GgufTensor* gt = gr.findTensor("blk.0.ffn_gate_exps.weight");
        if (gt && gt->dimensions.size() == 3) {
            const std::uint64_t inDim  = gt->dimensions[0];
            const std::uint64_t outDim = gt->dimensions[1];
            const std::uint64_t nExp   = gt->dimensions[2];
            const std::uint64_t perExpert = inDim * outDim;
            std::printf("=== expert stacking: ffn_gate_exps [in=%llu out=%llu nExp=%llu] ===\n",
                        (unsigned long long)inDim, (unsigned long long)outDim,
                        (unsigned long long)nExp);
            // Dequant the whole stacked GGUF tensor once.
            const std::uint8_t* payload = mm.base() + gr.tensorDataOffset() + gt->fileOffset;
            std::vector<float> full(static_cast<std::size_t>(gt->nelements));
            mimirmind::compute::dequantToF32(gt->type, payload, gt->nelements, full.data());
            for (std::uint64_t E : {std::uint64_t(0), std::uint64_t(1), std::uint64_t(5), nExp - 1}) {
                const std::string hf = "model.language_model.layers.0.mlp.experts."
                                       + std::to_string(E) + ".gate_proj";
                std::vector<float> a = dequantNvfp4Side(sm, hf);
                if (a.size() > 4096) a.resize(4096);
                std::vector<float> b(full.begin() + static_cast<std::ptrdiff_t>(E * perExpert),
                                     full.begin() + static_cast<std::ptrdiff_t>(E * perExpert + 4096));
                compare(("expert " + std::to_string(E) + " gate_proj").c_str(), a, b);
            }
            std::printf("\n");
        }
    }

    // ssm_a special: NVFP4 side is A_log passthrough -> apply -exp; must match GGUF ssm_a.
    {
        std::vector<float> a = dequantNvfp4Side(sm, "model.language_model.layers.0.linear_attn.A_log");
        // A_log has no ".weight" suffix — try the raw name too.
        if (a.empty()) {
            const auto* w = sm.find("model.language_model.layers.0.linear_attn.A_log");
            if (w) {
                const auto wb = sm.tensorBytes("model.language_model.layers.0.linear_attn.A_log");
                a.resize(static_cast<std::size_t>(w->nelements));
                const auto* h = reinterpret_cast<const std::uint16_t*>(wb.data());
                for (std::uint64_t i = 0; i < w->nelements; ++i) a[i] = bf16ToF32(h[i]);
            }
        }
        for (float& x : a) x = -std::exp(x);
        std::vector<float> b = dequantGgufSide(gr, mm, "blk.0.ssm_a", a.size());
        compare("L0 ssm_a  (-exp(A_log) vs GGUF SSM_A_NOSCAN)", a, b);
    }
    return 0;
}
