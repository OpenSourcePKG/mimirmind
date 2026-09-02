// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Pure-CPU unit tests for server::parseChatRequest (8.19 Increment 1):
// robustness helpers + OpenAI tool_choice object form + malformed->param.
// No GPU/model: parseChatRequest's only ChatTemplate.cpp dependency is the
// trivial `parseChatRole`, which is mirrored locally below so the test links
// without dragging in the Tokenizer/ChatTemplate encode machinery (every other
// model type it touches is header-only).

#include "TestFramework.hpp"

#include "server/ChatRequestParser.hpp"

#include <nlohmann/json.hpp>

#include <cctype>
#include <string>
#include <string_view>

// Local mirror of src/model/ChatTemplate.cpp::parseChatRole (the sole symbol
// ChatRequestParser.cpp references from that TU). Kept in sync intentionally.
namespace mimirmind::model {
bool parseChatRole(std::string_view s, ChatRole& out) noexcept {
    std::string low;
    low.reserve(s.size());
    for (char c : s) low.push_back(static_cast<char>(std::tolower(
        static_cast<unsigned char>(c))));
    if (low == "system")    { out = ChatRole::System;    return true; }
    if (low == "user")      { out = ChatRole::User;      return true; }
    if (low == "assistant") { out = ChatRole::Assistant; return true; }
    if (low == "tool")      { out = ChatRole::Tool;      return true; }
    return false;
}
} // namespace mimirmind::model

using ::mimirmind::server::ChatRequest;
using ::mimirmind::server::ChatRequestError;
using ::mimirmind::server::parseChatRequest;
using nlohmann::json;

namespace {

// A minimal valid single-user-message body other tests extend.
json baseBody() {
    return json{
        {"model", "qwen3.6"},
        {"messages", json::array({json{{"role", "user"}, {"content", "hi"}}})},
    };
}

// Assert that parsing `body` throws a ChatRequestError whose param == expected.
bool throwsWithParam(const json& body, std::string_view expectedParam) {
    try {
        (void)parseChatRequest(body);
    } catch (const ChatRequestError& e) {
        return e.param() == expectedParam;
    } catch (...) {
        return false;
    }
    return false;  // did not throw
}

} // namespace

// --- tool_choice object form (the original bug) ---------------------------

TEST(toolChoice_objectForm_forcesNamedCall) {
    json b = baseBody();
    b["tools"] = json::array({
        json{{"type", "function"}, {"function", {{"name", "get_weather"}}}},
        json{{"type", "function"}, {"function", {{"name", "send_email"}}}},
    });
    b["tool_choice"] = json{{"type", "function"},
                            {"function", {{"name", "get_weather"}}}};

    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.forcedToolName == "get_weather");
    EXPECT_TRUE(r.toolChoice == "required");         // reuses required-opener
    EXPECT_EQ(r.tools.size(), std::size_t{1});       // toolset narrowed
    EXPECT_TRUE(r.tools.at(0).name == "get_weather");
}

TEST(toolChoice_stringForms_stillWork) {
    for (const char* choice : {"auto", "none", "required"}) {
        json b = baseBody();
        b["tools"] = json::array({
            json{{"type", "function"}, {"function", {{"name", "a"}}}}});
        b["tool_choice"] = choice;
        const ChatRequest r = parseChatRequest(b);
        EXPECT_TRUE(r.toolChoice == choice);
        EXPECT_TRUE(r.forcedToolName.empty());
        EXPECT_EQ(r.tools.size(), std::size_t{1});   // not narrowed
    }
}

TEST(toolChoice_forceUnknownFunction_400paramToolChoice) {
    json b = baseBody();
    b["tools"] = json::array({
        json{{"type", "function"}, {"function", {{"name", "a"}}}}});
    b["tool_choice"] = json{{"type", "function"},
                            {"function", {{"name", "does_not_exist"}}}};
    EXPECT_TRUE(throwsWithParam(b, "tool_choice"));
}

TEST(toolChoice_objectMissingName_400paramToolChoice) {
    json b = baseBody();
    b["tool_choice"] = json{{"type", "function"}, {"function", json::object()}};
    EXPECT_TRUE(throwsWithParam(b, "tool_choice"));
}

// --- malformed value -> 400 with correct param ----------------------------

TEST(wrongType_temperature_400paramTemperature) {
    json b = baseBody();
    b["temperature"] = "hot";
    EXPECT_TRUE(throwsWithParam(b, "temperature"));
}

TEST(wrongType_maxTokens_400paramMaxTokens) {
    json b = baseBody();
    b["max_tokens"] = "lots";
    EXPECT_TRUE(throwsWithParam(b, "max_tokens"));
}

TEST(wrongType_stream_400paramStream) {
    json b = baseBody();
    b["stream"] = "yes";
    EXPECT_TRUE(throwsWithParam(b, "stream"));
}

TEST(wrongType_role_400paramRole) {
    json b = baseBody();
    b["messages"] = json::array({json{{"role", 42}, {"content", "hi"}}});
    EXPECT_TRUE(throwsWithParam(b, "messages[].role"));
}

TEST(missingMessages_400paramMessages) {
    json b = json{{"model", "qwen3.6"}};
    EXPECT_TRUE(throwsWithParam(b, "messages"));
}

TEST(contentArray_400paramContent_incr1) {
    // Increment 2 will concatenate text parts; Increment 1 keeps a clean,
    // field-tagged 400 (never an opaque type_error).
    json b = baseBody();
    b["messages"] = json::array({json{
        {"role", "user"},
        {"content", json::array({json{{"type", "text"}, {"text", "hi"}}})}}});
    EXPECT_TRUE(throwsWithParam(b, "messages[].content"));
}

// --- liberal accept: valid standard fields never 400 ----------------------

TEST(validSamplingAndLength_allHonored) {
    json b = baseBody();
    b["temperature"]           = 0.7;
    b["top_p"]                 = 0.9;
    b["top_k"]                 = 40;
    b["seed"]                  = 42;
    b["max_completion_tokens"] = 128;   // wins over max_tokens
    b["max_tokens"]            = 64;
    b["stream"]                = true;
    b["stop"]                  = json::array({"\n\n", "STOP"});
    b["frequency_penalty"]     = 0.1;
    b["presence_penalty"]      = 0.2;

    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.hasTemperature);
    EXPECT_NEAR(r.temperature, 0.7F, 1e-6F);
    EXPECT_EQ(r.topK, std::size_t{40});
    EXPECT_EQ(r.seed, std::uint64_t{42});
    // max_completion_tokens is parsed after max_tokens -> it wins.
    EXPECT_EQ(r.maxTokens, std::size_t{128});
    EXPECT_TRUE(r.stream);
    EXPECT_EQ(r.stopStrings.size(), std::size_t{2});
    EXPECT_TRUE(r.hasFrequencyPenalty);
    EXPECT_TRUE(r.hasPresencePenalty);
}

TEST(unknownAndIgnoredStandardFields_neverError) {
    json b = baseBody();
    // pure-IGNORE + unknown fields must be accepted, never 400.
    b["user"]          = "u-123";
    b["metadata"]      = json{{"k", "v"}};
    b["store"]         = true;
    b["service_tier"]  = "auto";
    b["modalities"]    = json::array({"text"});
    b["n"]             = 3;
    b["parallel_tool_calls"] = false;
    b["some_future_field"]   = 123;

    bool threw = false;
    try { (void)parseChatRequest(b); } catch (...) { threw = true; }
    EXPECT_TRUE(!threw);
}

TEST(contentNull_assistantToolCallOnly_ok) {
    // assistant turn with content:null + tool_calls must parse (multi-round).
    json b = baseBody();
    b["messages"] = json::array({
        json{{"role", "user"}, {"content", "weather?"}},
        json{{"role", "assistant"}, {"content", nullptr},
             {"tool_calls", json::array({json{
                 {"id", "call_1"}, {"type", "function"},
                 {"function", {{"name", "get_weather"},
                               {"arguments", "{\"city\":\"NYC\"}"}}}}})}},
        json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "sunny"}},
    });
    const ChatRequest r = parseChatRequest(b);
    EXPECT_EQ(r.messages.size(), std::size_t{3});
    EXPECT_EQ(r.messages.at(1).toolCalls.size(), std::size_t{1});
    EXPECT_TRUE(r.messages.at(2).toolCallId == "call_1");
}

int main() {
    return mm::test::run();
}
