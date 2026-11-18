// Hugin 7.2 Inc-1/Inc-2 — MCP server + inspect tools, driven in-process through
// the JSON-RPC layer against a synthesized GGUF (no fixture on disk, no GPU);
// the host-side logit math of the forward tools. The forward tools themselves
// run the engine and are validated end-to-end against a real model.

#include "TestFramework.hpp"

#include "core/rune/RuneFile.hpp"
#include "core/rune/RuneFormat.hpp"
#include "hugin/ForwardTools.hpp"
#include "hugin/GenerationCheck.hpp"
#include "hugin/RuneStaging.hpp"
#include "hugin/SurgeryTools.hpp"
#include "hugin/InspectTools.hpp"
#include "hugin/LogitMath.hpp"
#include "hugin/McpServer.hpp"
#include "hugin/SeparationMath.hpp"
#include "hugin/TreeParzenEstimator.hpp"
#include "compute/IRuneSteerOps.hpp"
#include "compute/cpu/GpuOps.hpp"
#include "core/cpu/CpuContext.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

using nlohmann::json;
namespace hg = mimirmind::hugin;
namespace rn = mimirmind::core::rune;

namespace {

bool nearD(double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(b)); }

/// GGUF v3: metadata general.alignment + general.architecture="testarch";
/// tensors "w" F32 dims [3,2] = {1..6}, "q" Q8_0 dims [32] (scale 0.5, ints -16..15).
std::string writeTestGguf() {
    std::vector<std::uint8_t> b;
    auto u32 = [&](std::uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i))); };
    auto u64 = [&](std::uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i))); };
    auto str = [&](std::string_view s) { u64(s.size()); for (char c : s) b.push_back(static_cast<std::uint8_t>(c)); };

    u32(0x46554747u); u32(3); u64(2); u64(2);
    str("general.alignment"); u32(4); u32(32);
    str("general.architecture"); u32(8); str("testarch");
    str("w"); u32(2); u64(3); u64(2); u32(0); u64(0);    // F32, 24 bytes
    str("q"); u32(1); u64(32); u32(8); u64(32);          // Q8_0 at offset 32, 34 bytes
    while (b.size() % 32 != 0) b.push_back(0);
    const float w[6] = {1, 2, 3, 4, 5, 6};
    const auto* wp = reinterpret_cast<const std::uint8_t*>(w);
    b.insert(b.end(), wp, wp + sizeof(w));
    while (b.size() % 32 != 0) b.push_back(0);
    b.push_back(0x00); b.push_back(0x38);                // f16 0.5
    for (int i = 0; i < 32; ++i) b.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(i - 16)));

    const std::string path = (std::filesystem::temp_directory_path() /
                              ("hugin_test_" + std::to_string(::getpid()) + ".gguf")).string();
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()),
                                               static_cast<std::streamsize>(b.size()));
    return path;
}

struct Fixture {
    std::string     path = writeTestGguf();
    hg::McpServer   server{"hugin", "test"};
    hg::InspectTools tools{path};
    hg::ForwardTools forwardTools{[this] { return hg::ForwardTools::Target{tools.modelPath(), {}}; }};
    hg::SurgeryTools surgery{forwardTools};
    hg::RuneStaging  staging{forwardTools, surgery,
                             std::filesystem::temp_directory_path() / "hugin_tests_runes", "test"};
    int             nextId{1};

    Fixture() {
        tools.registerOn(server);
        forwardTools.registerOn(server);
        surgery.registerOn(server);
        staging.registerOn(server);
    }
    ~Fixture() { std::filesystem::remove(path); }

    json rpc(const std::string& method, json params = json::object()) {
        auto r = server.handle({{"jsonrpc", "2.0"}, {"id", nextId++}, {"method", method},
                                {"params", std::move(params)}});
        return r ? *r : json(nullptr);
    }
    /// tools/call -> (isError, structured result or error text)
    std::pair<bool, json> call(const std::string& tool, json args = json::object()) {
        const json r = rpc("tools/call", {{"name", tool}, {"arguments", std::move(args)}})["result"];
        const bool err = r.value("isError", false);
        return {err, err ? json(r["content"][0]["text"]) : r["structuredContent"]};
    }
};

std::string writeRune(const std::string& base, const std::string& target, const std::string& hash,
                      std::vector<float> factors) {
    rn::RuneOp op;
    op.kind = rn::RuneOpKind::Scale; op.target = target; op.tensorHash = hash;
    op.payload = "s"; op.axis = rn::RuneScaleAxis::Cols;
    rn::RuneManifest m;
    m.name = "t"; m.baseFingerprint = base; m.provenanceJson = "{}"; m.ops = {op};
    const auto bytes = rn::RuneFile{m, {{"s", std::move(factors)}}}.serialize();
    const std::string p = (std::filesystem::temp_directory_path() /
                           ("hugin_test_" + std::to_string(::getpid()) + "_" + target + ".rune")).string();
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()),
                                            static_cast<std::streamsize>(bytes.size()));
    return p;
}

} // namespace

TEST(mcp_initialize_and_toolsList) {
    Fixture f;
    const json init = f.rpc("initialize", {{"protocolVersion", "2025-03-26"}})["result"];
    EXPECT_EQ(init["protocolVersion"].get<std::string>(), std::string{"2025-03-26"});
    EXPECT_EQ(init["serverInfo"]["name"].get<std::string>(), std::string{"hugin"});
    EXPECT_TRUE(init["capabilities"].contains("tools"));
    const json tools = f.rpc("tools/list")["result"]["tools"];
    std::vector<std::string> names;
    for (const auto& t : tools) {
        names.push_back(t["name"]);
        EXPECT_TRUE(t.contains("inputSchema"));
    }
    std::sort(names.begin(), names.end());
    EXPECT_TRUE(names == (std::vector<std::string>{"ablate_preview", "activations", "attention_pattern", "causal_trace", "causality_check",
                                                   "condition_vector", "eval", "eval_preview", "expert_usage", "generate", "list_tensors", "logit_lens",
                                                   "model_info", "next_token", "open_model", "perplexity", "rune_check",
                                                   "rune_commit", "rune_compare", "rune_gate", "rune_inspect", "rune_list",
                                                   "rune_rollback", "rune_stage_ablate", "rune_stage_experts", "rune_stage_heads",
                                                   "rune_stage_steer", "rune_stage_steer_if",
                                                   "steer_if_preview", "steer_list", "steer_preview",
                                                   "steer_vector", "tensor_hash",
                                                   "tensor_stats", "tokenize"}));
}

TEST(mcp_protocolErrors) {
    Fixture f;
    EXPECT_EQ(f.rpc("bogus/method")["error"]["code"].get<int>(), -32601);
    EXPECT_TRUE(!f.server.handle({{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}}).has_value());
    const auto bad = f.server.handleLine("{not json");
    EXPECT_TRUE(bad.has_value());
    EXPECT_EQ(json::parse(*bad)["error"]["code"].get<int>(), -32700);
    EXPECT_EQ(f.rpc("tools/call", {{"name", "nope"}})["error"]["code"].get<int>(), -32602);
    EXPECT_TRUE(f.rpc("ping")["result"].is_object());
}

TEST(mcp_runLoop_lineProtocol) {
    Fixture f;
    std::istringstream in(R"({"jsonrpc":"2.0","id":1,"method":"ping"})" "\n\n"
                          R"({"jsonrpc":"2.0","method":"notifications/initialized"})" "\n"
                          R"({"jsonrpc":"2.0","id":2,"method":"tools/list"})" "\n");
    std::ostringstream out;
    f.server.run(in, out);
    std::istringstream lines(out.str());
    std::string l1, l2, l3;
    std::getline(lines, l1); std::getline(lines, l2);
    EXPECT_EQ(json::parse(l1)["id"].get<int>(), 1);
    EXPECT_EQ(json::parse(l2)["id"].get<int>(), 2);
    EXPECT_TRUE(!std::getline(lines, l3));   // notification produced no line
}

TEST(tools_modelInfo_and_listTensors) {
    Fixture f;
    const auto [err, info] = f.call("model_info");
    EXPECT_TRUE(!err);
    EXPECT_EQ(info["architecture"].get<std::string>(), std::string{"testarch"});
    EXPECT_EQ(info["tensor_count"].get<int>(), 2);
    EXPECT_EQ(info["types"]["Q8_0"]["bytes"].get<int>(), 34);
    const auto [err2, list] = f.call("list_tensors", {{"filter", "w"}});
    EXPECT_TRUE(!err2);
    EXPECT_EQ(list["matched"].get<int>(), 1);
    EXPECT_EQ(list["tensors"][0]["rows"].get<int>(), 2);
    EXPECT_EQ(list["tensors"][0]["cols"].get<int>(), 3);
    EXPECT_TRUE(list["tensors"][0]["rune_patchable"].get<bool>());
}

TEST(tools_tensorStats_f32_rowRange_and_q8) {
    Fixture f;
    const auto [e1, s] = f.call("tensor_stats", {{"name", "w"}});
    EXPECT_TRUE(!e1);
    EXPECT_TRUE(nearD(s["mean"].get<double>(), 3.5));
    EXPECT_TRUE(nearD(s["std"].get<double>(), std::sqrt(35.0 / 12.0)));
    EXPECT_TRUE(nearD(s["l2"].get<double>(), std::sqrt(91.0)));
    EXPECT_TRUE(nearD(s["row_norm"]["max"].get<double>(), std::sqrt(77.0)));   // row {4,5,6}
    const auto [e2, r] = f.call("tensor_stats", {{"name", "w"}, {"row_begin", 1}, {"row_end", 2}});
    EXPECT_TRUE(!e2);
    EXPECT_TRUE(nearD(r["mean"].get<double>(), 5.0));
    const auto [e3, q] = f.call("tensor_stats", {{"name", "q"}});
    EXPECT_TRUE(!e3);
    EXPECT_TRUE(nearD(q["min"].get<double>(), -8.0));   // 0.5 * -16
    EXPECT_TRUE(nearD(q["max"].get<double>(), 7.5));    // 0.5 * 15
    const auto [e4, msg] = f.call("tensor_stats", {{"name", "missing"}});
    EXPECT_TRUE(e4);
    EXPECT_TRUE(msg.get<std::string>().find("missing") != std::string::npos);
}

TEST(tools_runeCheck_dryRun_matchesPrediction_andNeverWrites) {
    Fixture f;
    const std::string base  = f.call("model_info").second["base_fingerprint"];
    const std::string hashW = f.call("tensor_hash", {{"name", "w"}}).second["tensor_hash"];

    const std::string ok = writeRune(base, "w", hashW, {2.0f, 2.0f, 2.0f});
    const auto [e1, r] = f.call("rune_check", {{"paths", {ok}}});
    EXPECT_TRUE(!e1);
    EXPECT_TRUE(r["ok"].get<bool>());
    const float doubled[6] = {2, 4, 6, 8, 10, 12};
    EXPECT_EQ(r["runes"][0]["touched"][0]["hash_after"].get<std::string>(),
              rn::tensorContentHash(doubled, sizeof(doubled)));
    // Read-only: the model still hashes to the original.
    EXPECT_EQ(f.call("tensor_hash", {{"name", "w"}}).second["tensor_hash"].get<std::string>(), hashW);

    const auto [e2, insp] = f.call("rune_inspect", {{"path", ok}});
    EXPECT_TRUE(!e2);
    EXPECT_EQ(insp["ops"][0]["axis"].get<std::string>(), std::string{"cols"});
    EXPECT_EQ(insp["ops"][0]["payload_values"].get<int>(), 3);

    // Q8_0 accepts only exact block scales (7.3); 0.3 is not exact in f16 and is refused.
    const std::string hashQ = f.call("tensor_hash", {{"name", "q"}}).second["tensor_hash"].get<std::string>();
    const std::string quant = writeRune(base, "q", hashQ, std::vector<float>(32, 0.3f));
    const auto [e3, rq] = f.call("rune_check", {{"paths", {quant}}});
    EXPECT_TRUE(!e3);
    EXPECT_TRUE(!rq["ok"].get<bool>());
    EXPECT_TRUE(rq["error"].get<std::string>().find("not exact") != std::string::npos);
    const std::string mask = writeRune(base, "q", hashQ, std::vector<float>(32, 0.0f));
    const auto [e5, rm] = f.call("rune_check", {{"paths", {mask}}});
    EXPECT_TRUE(!e5);
    EXPECT_TRUE(rm["ok"].get<bool>());

    const std::string wrongBase = writeRune("other-model", "w", hashW, {2.0f, 2.0f, 2.0f});
    const auto [e4, rb] = f.call("rune_check", {{"paths", {wrongBase}}});
    EXPECT_TRUE(!rb["ok"].get<bool>());
    EXPECT_TRUE(rb["error"].get<std::string>().find(base) != std::string::npos);

    for (const auto& p : {ok, quant, wrongBase}) std::filesystem::remove(p);
}

TEST(tools_withoutModel_reportToolError) {
    hg::McpServer    server{"hugin", "test"};
    hg::InspectTools tools{""};
    tools.registerOn(server);
    const json r = (*server.handle({{"jsonrpc", "2.0"}, {"id", 1}, {"method", "tools/call"},
                                    {"params", {{"name", "model_info"}}}}))["result"];
    EXPECT_TRUE(r["isError"].get<bool>());
    EXPECT_TRUE(r["content"][0]["text"].get<std::string>().find("open_model") != std::string::npos);
}

TEST(forwardTools_withoutModel_reportToolError_noEngineBuilt) {
    hg::McpServer    server{"hugin", "test"};
    hg::ForwardTools fwd{[] { return hg::ForwardTools::Target{}; }};
    fwd.registerOn(server);
    for (const char* tool : {"next_token", "perplexity", "logit_lens", "activations", "tokenize",
                             "attention_pattern", "expert_usage"}) {
        const json r = (*server.handle({{"jsonrpc", "2.0"}, {"id", 1}, {"method", "tools/call"},
                                        {"params", {{"name", tool}, {"arguments", {{"text", "hi"}}}}}}))["result"];
        EXPECT_TRUE(r["isError"].get<bool>());
        EXPECT_TRUE(r["content"][0]["text"].get<std::string>().find("open_model") != std::string::npos);
    }
}

TEST(surgeryTools_withoutModel_reportToolError_steerListEmpty) {
    hg::McpServer    server{"hugin", "test"};
    hg::ForwardTools fwd{[] { return hg::ForwardTools::Target{}; }};
    hg::SurgeryTools surgery{fwd};
    surgery.registerOn(server);
    const auto call = [&](const char* tool, json args) {
        return (*server.handle({{"jsonrpc", "2.0"}, {"id", 1}, {"method", "tools/call"},
                                {"params", {{"name", tool}, {"arguments", std::move(args)}}}}))["result"];
    };
    const json calls[] = {
        json{{"tool", "causal_trace"}, {"args", {{"clean", "a"}, {"corrupt", "b"}}}},
        json{{"tool", "steer_vector"}, {"args", {{"name", "v"}, {"positive", {"a"}}, {"negative", {"b"}}, {"block", 0}}}},
        json{{"tool", "steer_preview"}, {"args", {{"text", "hi"}, {"vector", "v"}}}},
        json{{"tool", "ablate_preview"}, {"args", {{"text", "hi"}, {"block", 0}, {"heads", {0}}}}},
        json{{"tool", "generate"}, {"args", {{"text", "hi"}}}},
        json{{"tool", "eval_preview"},
             {"args", {{"items", json::array({{{"prompt", "a"}, {"expect", "b"}}})},
                       {"head_edits", json::array({{{"block", 0}, {"heads", {0}}, {"scale", 2.0}}})}}}}};
    for (const auto& c : calls) {
        const json r = call(c["tool"].get<std::string>().c_str(), c["args"]);
        EXPECT_TRUE(r["isError"].get<bool>());
        EXPECT_TRUE(r["content"][0]["text"].get<std::string>().find("open_model") != std::string::npos);
    }
    const json list = call("steer_list", json::object());
    EXPECT_TRUE(!list.value("isError", false));
}

TEST(runeStaging_guards_withoutModel_andUnknownNames) {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("hugin_rs_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    hg::McpServer    server{"hugin", "test"};
    hg::ForwardTools fwd{[] { return hg::ForwardTools::Target{}; }};
    hg::SurgeryTools surgery{fwd};
    hg::RuneStaging  staging{fwd, surgery, dir, "test"};
    staging.registerOn(server);
    const auto call = [&](const char* tool, json args) {
        return (*server.handle({{"jsonrpc", "2.0"}, {"id", 1}, {"method", "tools/call"},
                                {"params", {{"name", tool}, {"arguments", std::move(args)}}}}))["result"];
    };
    const auto errText = [](const json& r) {
        return r["isError"].get<bool>() ? r["content"][0]["text"].get<std::string>() : std::string{};
    };
    EXPECT_TRUE(errText(call("rune_stage_steer", {{"name", "s"}, {"vector", "v"}, {"alpha", 1.0}}))
                    .find("open_model") != std::string::npos);
    EXPECT_TRUE(errText(call("rune_stage_steer_if", {{"name", "c"}, {"vector", "v"}, {"alpha", 1.0}}))
                    .find("open_model") != std::string::npos);
    EXPECT_TRUE(errText(call("rune_stage_ablate", {{"name", "a"}, {"block", 0}, {"heads", {0}}}))
                    .find("open_model") != std::string::npos);
    EXPECT_TRUE(errText(call("rune_stage_experts", {{"name", "e"}, {"block", 0}, {"experts", {0}}}))
                    .find("open_model") != std::string::npos);
    EXPECT_TRUE(errText(call("rune_stage_steer", {{"name", "../x"}, {"vector", "v"}, {"alpha", 1.0}}))
                    .find("rune name") != std::string::npos);
    EXPECT_TRUE(errText(call("rune_stage_steer", {{"name", "a/b"}, {"vector", "v"}, {"alpha", 1.0}}))
                    .find("[A-Za-z0-9._-]") != std::string::npos);
    EXPECT_TRUE(errText(call("rune_gate", {{"name", "nope"}, {"target", {{"items", json::array()}}}}))
                    .find("no rune named") != std::string::npos);
    EXPECT_TRUE(errText(call("rune_commit", {{"name", "nope"}})).find("no rune named") != std::string::npos);
    EXPECT_TRUE(errText(call("rune_rollback", {{"name", "nope"}})).find("no rune named") != std::string::npos);
    const json list = call("rune_list", json::object());
    EXPECT_TRUE(!list.value("isError", false));
    std::filesystem::remove_all(dir);
}

TEST(logitMath_logSoftmax_topK_rank_entropy_kl) {
    const std::vector<float> logits{1.0f, 3.0f, 2.0f, 3.0f};
    const auto lp = hg::LogitMath::logSoftmax(logits);
    double total = 0;
    for (double v : lp) total += std::exp(v);
    EXPECT_TRUE(nearD(total, 1.0));
    EXPECT_TRUE(nearD(lp[1] - lp[0], 2.0));
    EXPECT_TRUE(hg::LogitMath::topK(lp, 3) == (std::vector<std::int32_t>{1, 3, 2}));   // ties by index
    EXPECT_EQ(hg::LogitMath::rankOf(lp, 3), std::size_t{1});
    EXPECT_EQ(hg::LogitMath::rankOf(lp, 0), std::size_t{3});
    const std::vector<float> uniform(4, 0.5f);
    const auto lu = hg::LogitMath::logSoftmax(uniform);
    EXPECT_TRUE(nearD(hg::LogitMath::entropy(lu), std::log(4.0)));
    EXPECT_TRUE(nearD(hg::LogitMath::kl(lp, lp), 0.0));
    // KL(p||u) = log(n) - H(p)
    EXPECT_TRUE(nearD(hg::LogitMath::kl(lp, lu), std::log(4.0) - hg::LogitMath::entropy(lp)));
    std::vector<float> capped{100.0f, -100.0f, 0.0f};
    hg::LogitMath::softcap(capped, 30.0f);
    EXPECT_TRUE(capped[0] < 30.0f && capped[0] > 29.0f && capped[1] > -30.0f && capped[2] == 0.0f);
}

TEST(eval_withoutModel_andBadGoldset_reportToolError) {
    hg::McpServer    server{"hugin", "test"};
    hg::ForwardTools fwd{[] { return hg::ForwardTools::Target{}; }};
    fwd.registerOn(server);
    const auto call = [&](json args) {
        return (*server.handle({{"jsonrpc", "2.0"}, {"id", 1}, {"method", "tools/call"},
                                {"params", {{"name", "eval"}, {"arguments", std::move(args)}}}}))["result"];
    };
    // Goldset problems are reported before any engine is needed.
    const json noSet = call(json::object());
    EXPECT_TRUE(noSet["isError"].get<bool>());
    EXPECT_TRUE(noSet["content"][0]["text"].get<std::string>().find("items") != std::string::npos);
    const json missing = call({{"path", "/nonexistent/goldset.jsonl"}});
    EXPECT_TRUE(missing["content"][0]["text"].get<std::string>().find("cannot open") != std::string::npos);
    const json noModel = call({{"items", {{{"prompt", "a"}, {"expect", " b"}}}}});
    EXPECT_TRUE(noModel["content"][0]["text"].get<std::string>().find("open_model") != std::string::npos);
}

TEST(generationCheck_language_mixing_and_facts) {
    using GC = hg::GenerationCheck;
    // Clean answers in each language.
    const GC de{json{{"lang", "de"}, {"match", "Kilimanjaro.*5[.,]?89\\d"}}};
    const json okDe = de.evaluate("Der höchste Berg Afrikas ist der Kilimanjaro mit 5.895 Metern.");
    EXPECT_TRUE(okDe["correct"].get<bool>());
    EXPECT_EQ(okDe["lang"].get<std::string>(), std::string{"de"});
    // English answer to a German instruction: language fails, the fact still holds.
    const json en = de.evaluate("The tallest mountain in Africa is Kilimanjaro,\nat 5,895 meters.");
    EXPECT_TRUE(!en["checks"]["lang"].get<bool>());
    EXPECT_TRUE(en["checks"]["match"].get<bool>());   // newline is matched as a space
    // Fact drift (attempt-2 canary).
    const json drift = de.evaluate("Der höchste Berg Afrikas ist der Kilimanjaro mit 5.199 Metern.");
    EXPECT_TRUE(drift["checks"]["lang"].get<bool>());
    EXPECT_TRUE(!drift["checks"]["match"].get<bool>());
    EXPECT_TRUE(!drift["correct"].get<bool>());
    // Code-mixing in an English answer (the attempt-2 failure).
    const GC enc{json{{"lang", "en"}, {"match", "Portug"}}};
    const json mixed = enc.evaluate("In Brazil wird Portugiesisch gesprochen.");
    EXPECT_TRUE(!mixed["checks"]["no_mixing"].get<bool>());
    EXPECT_TRUE(!mixed["correct"].get<bool>());
    const json clean = enc.evaluate("The official language of Brazil is Portuguese.");
    EXPECT_TRUE(clean["correct"].get<bool>());
    // A single foreign marker is tolerated by default, none with max_other_markers 0.
    EXPECT_TRUE(enc.evaluate("The capital is called Brasília, and Portuguese is spoken there; "
                             "the word is das.")["checks"]["no_mixing"].get<bool>());
    const GC strict{json{{"lang", "en"}, {"max_other_markers", 0}}};
    EXPECT_TRUE(!strict.evaluate("The language is Portuguese, das ist klar.")["checks"]["no_mixing"].get<bool>());
    // no_match and case-insensitivity.
    const GC nm{json{{"match", "ottawa"}, {"no_match", json::array({"toronto"})}}};
    EXPECT_TRUE(nm.evaluate("The capital of Canada is Ottawa.")["correct"].get<bool>());
    EXPECT_TRUE(!nm.evaluate("Ottawa, not Toronto.")["correct"].get<bool>());
    // Bad specs are refused with the field name.
    const auto refuses = [](json spec, const std::string& what) {
        try {
            (void)GC{spec};
        } catch (const std::exception& ex) {
            return std::string{ex.what()}.find(what) != std::string::npos;
        }
        return false;
    };
    EXPECT_TRUE(refuses({{"lang", "fr"}}, "lang"));
    EXPECT_TRUE(refuses({{"match", "("}}, "bad regex"));
    EXPECT_TRUE(refuses({{"max_tokens", 0}, {"lang", "de"}}, "max_tokens"));
    EXPECT_TRUE(refuses(json::object(), "at least one"));
    const GC dflt{json{{"lang", "de"}}};
    EXPECT_EQ(dflt.maxTokens(), GC::kDefaultMaxTokens);
}

TEST(eval_badGenerateSpec_reportedBeforeAnyEngine) {
    hg::McpServer    server{"hugin", "test"};
    hg::ForwardTools fwd{[] { return hg::ForwardTools::Target{}; }};
    fwd.registerOn(server);
    const json r = (*server.handle({{"jsonrpc", "2.0"}, {"id", 1}, {"method", "tools/call"},
                                    {"params", {{"name", "eval"},
                                                {"arguments", {{"items", {{{"prompt", "a"},
                                                    {"generate", {{"lang", "de"}, {"match", "(["}}}}}}}}}}}))["result"];
    EXPECT_TRUE(r["isError"].get<bool>());
    const std::string text = r["content"][0]["text"].get<std::string>();
    EXPECT_TRUE(text.find("goldset item 0") != std::string::npos);
    EXPECT_TRUE(text.find("bad regex") != std::string::npos);
    EXPECT_TRUE(text.find("open_model") == std::string::npos);
}

TEST(openModel_checkpointDir_forwardOnly) {
    Fixture f;
    const std::string dir = (std::filesystem::temp_directory_path() /
                             ("hugin_ckpt_" + std::to_string(::getpid()))).string();
    std::filesystem::create_directories(dir);
    const auto [e1, r] = f.call("open_model", {{"path", dir}, {"tokenizer_gguf", "/x/tok.gguf"}});
    EXPECT_TRUE(!e1);
    EXPECT_EQ(r["format"].get<std::string>(), std::string{"nvfp4"});
    EXPECT_EQ(f.tools.modelPath(), dir);
    EXPECT_EQ(f.tools.tokenizerGguf(), std::string{"/x/tok.gguf"});
    const auto [e2, msg] = f.call("model_info");
    EXPECT_TRUE(e2);
    EXPECT_TRUE(msg.get<std::string>().find("GGUF-only") != std::string::npos);
    // Back to a GGUF: inspection works again and the tokenizer override is cleared.
    const auto [e3, back] = f.call("open_model", {{"path", f.path}});
    EXPECT_TRUE(!e3);
    EXPECT_EQ(back["format"].get<std::string>(), std::string{"gguf"});
    EXPECT_TRUE(f.tools.tokenizerGguf().empty());
    std::filesystem::remove_all(dir);
}

// 7.11 — the CPU reference of the projection / conditional steer ops.
TEST(runeSteerOps_cpu_rowDot_gate_and_projection) {
    mimirmind::core::cpu::CpuContext ctx;
    mimirmind::compute::cpu::GpuOps  ops{ctx};
    mimirmind::compute::IRuneSteerOps* rs = ops.runeSteerOps();
    EXPECT_TRUE(rs != nullptr);
    constexpr std::size_t d = 4;
    // 3 rows: aligned with c, orthogonal to c, zero.
    std::vector<float> x{2, 0, 0, 0,   0, 3, 4, 0,   0, 0, 0, 0};
    const std::vector<float> c{1, 0, 0, 0};
    std::vector<float> score(3, -9.0F);
    rs->rowDotAsync(x.data(), 3, d, c.data(), score.data(), /*normalize=*/true);
    EXPECT_TRUE(score == (std::vector<float>{1.0F, 0.0F, 0.0F}));
    rs->rowDotAsync(x.data(), 3, d, c.data(), score.data(), /*normalize=*/false);
    EXPECT_TRUE(score == (std::vector<float>{2.0F, 0.0F, 0.0F}));
    // Gated add: only row 0 (cos 1 > 0.5) changes.
    rs->rowDotAsync(x.data(), 3, d, c.data(), score.data(), true);
    const std::vector<float> v{0, 0, 0, 1};
    std::vector<float> y = x;
    rs->steerRowsAsync(y.data(), 3, d, v.data(), 2.0F, score.data(), 0.5F, nullptr);
    EXPECT_TRUE(y == (std::vector<float>{2, 0, 0, 2,   0, 3, 4, 0,   0, 0, 0, 0}));
    // Ungated projection with alpha 1 removes the component along the unit
    // direction exactly: x -= (x.u) u.
    const std::vector<float> u{0, 0.6F, 0.8F, 0};
    std::vector<float> coef(3);
    std::vector<float> z = x;
    rs->rowDotAsync(z.data(), 3, d, u.data(), coef.data(), false);
    rs->steerRowsAsync(z.data(), 3, d, u.data(), -1.0F, nullptr, 0.0F, coef.data());
    std::vector<float> after(3);
    rs->rowDotAsync(z.data(), 3, d, u.data(), after.data(), false);
    for (const float a : after) {
        EXPECT_TRUE(std::fabs(a) < 1e-6F);
    }
    EXPECT_TRUE(z[0] == 2.0F);   // row 0 had no component along u
}

TEST(separationMath_auc_report_null) {
    using mimirmind::hugin::SeparationMath;
    // Perfect, inverted, tied.
    EXPECT_TRUE(std::fabs((SeparationMath::auc(std::vector<double>{3, 4}, std::vector<double>{1, 2})) - (1.0)) < 1e-12);
    EXPECT_TRUE(std::fabs((SeparationMath::auc(std::vector<double>{1, 2}, std::vector<double>{3, 4})) - (0.0)) < 1e-12);
    EXPECT_TRUE(std::fabs((SeparationMath::auc(std::vector<double>{1, 1}, std::vector<double>{1, 1})) - (0.5)) < 1e-12);
    // pos {1,3}, neg {2}: one of two pairs ordered -> 0.5; empty class -> 0.5.
    EXPECT_TRUE(std::fabs((SeparationMath::auc(std::vector<double>{1, 3}, std::vector<double>{2})) - (0.5)) < 1e-12);
    EXPECT_TRUE(std::fabs((SeparationMath::auc(std::vector<double>{}, std::vector<double>{2})) - (0.5)) < 1e-12);

    const auto r = SeparationMath::report(std::vector<double>{0.9, 0.2}, std::vector<double>{0.1, 0.6, 0.0, 0.0}, 0.5);
    EXPECT_TRUE(std::fabs((r.accuracy) - (4.0 / 6.0)) < 1e-12);          // 0.9 on, 0.2 off; 0.6 wrongly on
    EXPECT_TRUE(std::fabs((r.negativeFireRate) - (0.25)) < 1e-12);
    EXPECT_EQ(r.positives, std::size_t{2});
    EXPECT_EQ(r.negatives, std::size_t{4});

    // Permutation is a deterministic bijection.
    const auto p1 = SeparationMath::permutation(50, 7), p2 = SeparationMath::permutation(50, 7);
    EXPECT_TRUE(p1 == p2);
    auto sorted = p1;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < sorted.size(); ++i) EXPECT_EQ(sorted[i], i);

    const auto st = SeparationMath::summarize({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    EXPECT_TRUE(std::fabs((st.mean) - (0.55)) < 1e-12);
    EXPECT_TRUE(std::fabs((st.p95) - (1.0)) < 1e-12);
    EXPECT_TRUE(std::fabs((st.max) - (1.0)) < 1e-12);
    EXPECT_EQ(st.permutations, std::size_t{20});
}

TEST(separationMath_meanDifference_null_vs_signal) {
    using mimirmind::hugin::SeparationMath;
    // d >> n pure noise: the in-sample mean difference separates PERFECTLY even
    // for random labels — exactly why the null is needed.
    constexpr std::size_t d = 512, n = 16;
    std::vector<float> noise(n * d);
    std::uint64_t s = 1;
    for (auto& x : noise) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x = static_cast<float>(static_cast<double>(s >> 11U) / 9007199254740992.0 - 0.5);
    }
    std::vector<std::uint8_t> lab(n, 0);
    for (std::size_t i = 0; i < n / 2; ++i) lab[i] = 1;
    const double real = SeparationMath::meanDifferenceAuc(noise, d, lab);
    const auto   null = SeparationMath::meanDifferenceNull(noise, d, n / 2, 20, 99);
    EXPECT_TRUE(std::fabs((real) - (1.0)) < 1e-12);
    EXPECT_TRUE(std::fabs((null.p95) - (1.0)) < 1e-12);       // noise: real is NOT above the null

    // Real signal in a low dimension: real AUC 1, the null clearly below.
    constexpr std::size_t d2 = 2, n2 = 40;
    std::vector<float> sig(n2 * d2);
    std::vector<std::uint8_t> lab2(n2, 0);
    for (std::size_t i = 0; i < n2; ++i) {
        lab2[i] = i < n2 / 2 ? 1 : 0;
        sig[i * d2]     = (lab2[i] ? 1.0F : -1.0F) + 0.01F * static_cast<float>(i % 7);
        sig[i * d2 + 1] = 0.3F * static_cast<float>((i * 37) % 11) - 1.5F;
    }
    // The positives are the first n2 / 2 rows, as meanDifferenceNull expects.
    EXPECT_TRUE(std::fabs((SeparationMath::meanDifferenceAuc(sig, d2, lab2)) - (1.0)) < 1e-12);
    const auto null2 = SeparationMath::meanDifferenceNull(sig, d2, n2 / 2, 50, 3);
    EXPECT_TRUE(null2.p95 < 0.95);
}

TEST(surgery_causalityKeys_canonical) {
    using hg = mimirmind::hugin::SurgeryTools;
    // Order of edits, order and duplicates of heads do not matter; scale and block do.
    const auto k1 = hg::headsKey({{12, {3, 1, 2}, 2.0}, {4, {7}, 0.0}});
    const auto k2 = hg::headsKey({{4, {7, 7}, 0.0}, {12, {2, 3, 1}, 2.0}});
    EXPECT_TRUE(k1 == k2);
    EXPECT_TRUE(k1 != hg::headsKey({{12, {1, 2, 3}, 4.0}, {4, {7}, 0.0}}));
    EXPECT_TRUE(k1 != hg::headsKey({{13, {1, 2, 3}, 2.0}, {4, {7}, 0.0}}));
    EXPECT_TRUE(hg::vectorKey("fr", 1.0) == hg::vectorKey("fr", 1.0));
    EXPECT_TRUE(hg::vectorKey("fr", 1.0) != hg::vectorKey("fr", 0.5));
    EXPECT_TRUE(hg::vectorKey("fr", 1.0) != hg::vectorKey("de", 1.0));
}

TEST(tpe_pareto_and_rank) {
    using T = mimirmind::hugin::TreeParzenEstimator;
    std::vector<T::Trial> tr{{{0}, {1, 5}, true}, {{0}, {2, 2}, true}, {{0}, {3, 3}, true},
                             {{0}, {5, 1}, true}, {{0}, {0, 0}, false}};
    const auto front = T::paretoFront(tr);
    EXPECT_TRUE(front == (std::vector<std::size_t>{0, 1, 3}));   // infeasible (0,0) excluded
    const auto order = T::rank(tr);
    EXPECT_TRUE(order.back() == 4);                                 // infeasible last
    EXPECT_TRUE(order[3] == 2);                                     // dominated by (2,2)
}

TEST(tpe_minimises_quadratic_better_than_random) {
    using T = mimirmind::hugin::TreeParzenEstimator;
    using K = T::Param::Kind;
    const std::vector<T::Param> space{{"x", K::Float, -5, 5}, {"k", K::Int, 1, 64, true},
                                      {"c", K::Choice, 0, 0, false, 4}};
    // Optimum x = 1.5, k = 16, c = 2.
    const auto f = [](const std::vector<double>& p) {
        const double lk = std::log2(p[1]) - 4.0;
        return (p[0] - 1.5) * (p[0] - 1.5) + lk * lk + (p[2] == 2.0 ? 0.0 : 2.0);
    };
    double tpeBest = 0, rndBest = 0;
    for (std::uint64_t seed = 1; seed <= 5; ++seed) {
        T tpe(space, 1, {8, 0.25, 24, seed});
        T rnd(space, 1, {1000, 0.25, 24, seed});   // never leaves the startup phase
        double bt = 1e9, br = 1e9;
        for (int i = 0; i < 40; ++i) {
            auto p = tpe.suggest();
            EXPECT_TRUE(p[1] >= 1 && p[1] <= 64 && p[1] == std::round(p[1]));
            EXPECT_TRUE(p[2] >= 0 && p[2] <= 3 && p[2] == std::round(p[2]));
            const double y = f(p);
            bt = std::min(bt, y);
            tpe.tell(std::move(p), {y}, true);
            auto q = rnd.suggest();
            const double z = f(q);
            br = std::min(br, z);
            rnd.tell(std::move(q), {z}, true);
        }
        tpeBest += bt;
        rndBest += br;
        EXPECT_TRUE(bt < 1.0);
    }
    EXPECT_TRUE(tpeBest < 0.5 * rndBest);   // summed over the five seeds
    // Determinism: same seed, same suggestions.
    T a(space, 1, {2, 0.25, 24, 42}), b(space, 1, {2, 0.25, 24, 42});
    for (int i = 0; i < 6; ++i) {
        const auto pa = a.suggest(), pb = b.suggest();
        EXPECT_TRUE(pa == pb);
        a.tell(pa, {f(pa)}, true);
        b.tell(pb, {f(pb)}, true);
    }
}

int main() {
    return mm::test::run();
}
