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

// (content-array behavior is covered by the Increment 2 contentParts_* tests
//  below — arrays of text parts are now concatenated, not rejected.)

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

// --- Increment 2: content parts, developer role, deprecated functions,
//     response_format --------------------------------------------------------

TEST(contentParts_textArray_concatenated) {
    json b = baseBody();
    b["messages"] = json::array({json{
        {"role", "user"},
        {"content", json::array({
            json{{"type", "text"}, {"text", "Hello "}},
            json{{"type", "text"}, {"text", "world"}},
        })}}});
    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.messages.at(0).content == "Hello world");
}

TEST(contentParts_ignoresImageParts_noError) {
    json b = baseBody();
    b["messages"] = json::array({json{
        {"role", "user"},
        {"content", json::array({
            json{{"type", "text"}, {"text", "look:"}},
            json{{"type", "image_url"},
                 {"image_url", {{"url", "data:image/png;base64,AAAA"}}}},
        })}}});
    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.messages.at(0).content == "look:");   // image part ignored
}

TEST(contentScalarNumber_400paramContent) {
    json b = baseBody();
    b["messages"] = json::array({json{{"role", "user"}, {"content", 42}}});
    EXPECT_TRUE(throwsWithParam(b, "messages[].content"));
}

TEST(developerRole_mappedToSystem) {
    json b = baseBody();
    b["messages"] = json::array({
        json{{"role", "developer"}, {"content", "be terse"}},
        json{{"role", "user"}, {"content", "hi"}}});
    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.messages.at(0).role == ::mimirmind::model::ChatRole::System);
}

TEST(deprecatedFunctions_mappedToTools) {
    json b = baseBody();
    b["functions"] = json::array({
        json{{"name", "get_weather"}, {"description", "w"},
             {"parameters", json::object()}}});
    const ChatRequest r = parseChatRequest(b);
    EXPECT_EQ(r.tools.size(), std::size_t{1});
    EXPECT_TRUE(r.tools.at(0).name == "get_weather");
}

TEST(deprecatedFunctionCall_objectForcesNamedCall) {
    json b = baseBody();
    b["functions"] = json::array({
        json{{"name", "get_weather"}}, json{{"name", "other"}}});
    b["function_call"] = json{{"name", "get_weather"}};
    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.forcedToolName == "get_weather");
    EXPECT_TRUE(r.toolChoice == "required");
    EXPECT_EQ(r.tools.size(), std::size_t{1});     // narrowed to the forced fn
}

TEST(deprecatedFunctionCall_stringNone) {
    json b = baseBody();
    b["functions"]     = json::array({json{{"name", "a"}}});
    b["function_call"] = "none";
    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.toolChoice == "none");
    EXPECT_TRUE(r.forcedToolName.empty());
}

TEST(modernToolsWin_overDeprecatedFunctions) {
    json b = baseBody();
    b["tools"] = json::array({
        json{{"type", "function"}, {"function", {{"name", "modern"}}}}});
    b["functions"] = json::array({json{{"name", "legacy"}}});
    const ChatRequest r = parseChatRequest(b);
    EXPECT_EQ(r.tools.size(), std::size_t{1});
    EXPECT_TRUE(r.tools.at(0).name == "modern");    // functions ignored
}

TEST(responseFormat_types_parsed) {
    {
        json b = baseBody();
        b["response_format"] = json{{"type", "text"}};
        EXPECT_TRUE(parseChatRequest(b).responseFormat ==
                    ::mimirmind::server::ResponseFormat::Text);
    }
    {
        json b = baseBody();
        b["response_format"] = json{{"type", "json_object"}};
        EXPECT_TRUE(parseChatRequest(b).responseFormat ==
                    ::mimirmind::server::ResponseFormat::JsonObject);
    }
    {
        json b = baseBody();
        b["response_format"] = json{
            {"type", "json_schema"},
            {"json_schema", {{"name", "x"}, {"schema", json::object()}}}};
        EXPECT_TRUE(parseChatRequest(b).responseFormat ==
                    ::mimirmind::server::ResponseFormat::JsonSchema);
    }
}

TEST(responseFormat_malformed_400paramResponseFormat) {
    json b = baseBody();
    b["response_format"] = 7;   // not an object
    EXPECT_TRUE(throwsWithParam(b, "response_format"));

    json b2 = baseBody();
    b2["response_format"] = json{{"type", "yaml"}};   // unknown type
    EXPECT_TRUE(throwsWithParam(b2, "response_format"));
}

// --- Increment 3: stream_options.include_usage, reasoning_effort -----------

TEST(streamOptions_includeUsage_parsed) {
    json b = baseBody();
    b["stream"]         = true;
    b["stream_options"] = json{{"include_usage", true}};
    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.includeUsage);
}

TEST(streamOptions_absent_defaultsFalse) {
    const ChatRequest r = parseChatRequest(baseBody());
    EXPECT_TRUE(!r.includeUsage);
}

TEST(streamOptions_notObject_400param) {
    json b = baseBody();
    b["stream_options"] = "usage";
    EXPECT_TRUE(throwsWithParam(b, "stream_options"));
}

TEST(reasoningEffort_high_enablesThinking) {
    json b = baseBody();
    b["reasoning_effort"] = "high";
    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.enableThinking.has_value());
    EXPECT_TRUE(r.enableThinking.value());
}

TEST(reasoningEffort_none_disablesThinking) {
    json b = baseBody();
    b["reasoning_effort"] = "none";
    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.enableThinking.has_value());
    EXPECT_TRUE(!r.enableThinking.value());
}

TEST(reasoningEffort_doesNotOverrideExplicitEnableThinking) {
    json b = baseBody();
    b["enable_thinking"]  = false;   // explicit wins
    b["reasoning_effort"] = "high";
    const ChatRequest r = parseChatRequest(b);
    EXPECT_TRUE(r.enableThinking.has_value());
    EXPECT_TRUE(!r.enableThinking.value());
}

int main() {
    return mm::test::run();
}
