// Pure-CPU tests for the M-Munin wire-op layer. Covers the JSON
// envelope serde for request / healthz / error frames. The socket-
// integration layer (AttachSession + SocketServer) is exercised in
// the docker-based smoke tests that come with Schritt 8 — nothing
// meaningful to unit-test here without a live L0 device and a peer
// worker.
//
// Run via: cmake --build build --target munin_tests && build/munin_tests

#include "TestFramework.hpp"

#include "core/ipc/WireOps.hpp"

#include <string>

using ::mimirmind::core::ipc::HealthzResponse;
using ::mimirmind::core::ipc::ModelSummaryWire;
using ::mimirmind::core::ipc::RequestEnvelope;
using ::mimirmind::core::ipc::makeErrorJson;
using ::mimirmind::core::ipc::parseErrorJson;

// ---- RequestEnvelope --------------------------------------------------------

TEST(requestEnvelope_healthz_parses) {
    const auto r = RequestEnvelope::fromJson(R"({"op":"healthz"})");
    EXPECT_TRUE(static_cast<bool>(r));
    EXPECT_EQ(r->op, std::string{"healthz"});
    EXPECT_EQ(r->modelId, std::string{""});
}

TEST(requestEnvelope_attach_withModelId_parses) {
    const auto r = RequestEnvelope::fromJson(
        R"({"op":"attach","modelId":"gemma-4-e4b"})");
    EXPECT_TRUE(static_cast<bool>(r));
    EXPECT_EQ(r->op, std::string{"attach"});
    EXPECT_EQ(r->modelId, std::string{"gemma-4-e4b"});
}

TEST(requestEnvelope_missingOp_rejected) {
    const auto r = RequestEnvelope::fromJson(R"({"modelId":"x"})");
    EXPECT_TRUE(!r);
}

TEST(requestEnvelope_modelIdWrongType_rejected) {
    const auto r = RequestEnvelope::fromJson(
        R"({"op":"attach","modelId":42})");
    EXPECT_TRUE(!r);
}

TEST(requestEnvelope_malformedJson_rejected) {
    const auto r = RequestEnvelope::fromJson(R"({"op":)");
    EXPECT_TRUE(!r);
}

TEST(requestEnvelope_rootNotObject_rejected) {
    const auto r = RequestEnvelope::fromJson(R"("healthz")");
    EXPECT_TRUE(!r);
}

// ---- HealthzResponse round-trip --------------------------------------------

TEST(healthzResponse_roundtrip_emptyModels) {
    HealthzResponse r{};
    r.pid           = 4711;
    r.governorOwner = "munin";

    const std::string s = r.toJson();
    const auto parsed = HealthzResponse::fromJson(s);
    EXPECT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(parsed->protocolVersion, 1U);
    EXPECT_EQ(parsed->status, std::string{"ok"});
    EXPECT_EQ(parsed->governorOwner, std::string{"munin"});
    EXPECT_EQ(parsed->pid, 4711U);
    EXPECT_EQ(parsed->models.size(), 0U);
}

TEST(healthzResponse_roundtrip_twoModels) {
    HealthzResponse r{};
    r.pid = 123;
    {
        ModelSummaryWire m{};
        m.id          = "gemma-4-e4b";
        m.fingerprint = "720.25169342464.deadbeefcafebabe";
        m.totalBytes  = 25169342464ULL;
        m.tensorCount = 720;
        r.models.push_back(std::move(m));
    }
    {
        ModelSummaryWire m{};
        m.id          = "qwen2-7b";
        m.fingerprint = "512.7000000000.1234567890abcdef";
        m.totalBytes  = 7000000000ULL;
        m.tensorCount = 512;
        r.models.push_back(std::move(m));
    }

    const auto parsed = HealthzResponse::fromJson(r.toJson());
    EXPECT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(parsed->models.size(), 2U);
    EXPECT_EQ(parsed->models[0].id, std::string{"gemma-4-e4b"});
    EXPECT_EQ(parsed->models[0].tensorCount, 720U);
    EXPECT_EQ(parsed->models[0].totalBytes, 25169342464ULL);
    EXPECT_EQ(parsed->models[1].id, std::string{"qwen2-7b"});
    EXPECT_EQ(parsed->models[1].fingerprint,
              std::string{"512.7000000000.1234567890abcdef"});
}

TEST(healthzResponse_missingField_rejected) {
    // Drop `pid` — fromJson must refuse rather than default.
    const auto parsed = HealthzResponse::fromJson(
        R"({"protocol_version":1,"status":"ok","governor_owner":"munin","models":[]})");
    EXPECT_TRUE(!parsed);
}

// ---- Error envelope --------------------------------------------------------

TEST(errorEnvelope_roundtrip) {
    const std::string body = makeErrorJson("no model with id 'x' loaded");
    const auto parsed = parseErrorJson(body);
    EXPECT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(*parsed, std::string{"no model with id 'x' loaded"});
}

TEST(errorEnvelope_notAnError_rejected) {
    const auto parsed = parseErrorJson(R"({"status":"ok"})");
    EXPECT_TRUE(!parsed);
}

TEST(errorEnvelope_wrongType_rejected) {
    const auto parsed = parseErrorJson(R"({"error":42})");
    EXPECT_TRUE(!parsed);
}

int main() {
    return mm::test::run();
}