// Hugin 7.2 Inc-1/Inc-2 — MCP server + inspect tools, driven in-process through
// the JSON-RPC layer against a synthesized GGUF (no fixture on disk, no GPU);
// the host-side logit math of the forward tools. The forward tools themselves
// run the engine and are validated end-to-end against a real model.

#include "TestFramework.hpp"

#include "core/rune/RuneFile.hpp"
#include "core/rune/RuneFormat.hpp"
#include "hugin/ForwardTools.hpp"
#include "hugin/InspectTools.hpp"
#include "hugin/LogitMath.hpp"
#include "hugin/McpServer.hpp"

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
    hg::ForwardTools forwardTools{[this] { return tools.modelPath(); }};
    int             nextId{1};

    Fixture() {
        tools.registerOn(server);
        forwardTools.registerOn(server);
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
    EXPECT_TRUE(names == (std::vector<std::string>{"activations", "list_tensors", "logit_lens", "model_info",
                                                   "next_token", "open_model", "perplexity", "rune_check",
                                                   "rune_compare", "rune_inspect", "tensor_hash", "tensor_stats",
                                                   "tokenize"}));
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

    const std::string quant = writeRune(base, "q", "fnv1a64:0000000000000000", std::vector<float>(32, 1.0f));
    const auto [e3, rq] = f.call("rune_check", {{"paths", {quant}}});
    EXPECT_TRUE(!e3);
    EXPECT_TRUE(!rq["ok"].get<bool>());
    EXPECT_TRUE(rq["error"].get<std::string>().find("quantised") != std::string::npos);

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
    hg::ForwardTools fwd{[] { return std::string{}; }};
    fwd.registerOn(server);
    for (const char* tool : {"next_token", "perplexity", "logit_lens", "activations", "tokenize"}) {
        const json r = (*server.handle({{"jsonrpc", "2.0"}, {"id", 1}, {"method", "tools/call"},
                                        {"params", {{"name", tool}, {"arguments", {{"text", "hi"}}}}}}))["result"];
        EXPECT_TRUE(r["isError"].get<bool>());
        EXPECT_TRUE(r["content"][0]["text"].get<std::string>().find("open_model") != std::string::npos);
    }
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

int main() {
    return mm::test::run();
}
