// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/McpServer.hpp"

#include <algorithm>
#include <array>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mimirmind::hugin {

namespace {

using nlohmann::json;

/// MCP revisions this server implements, newest first. It only uses
/// initialize / ping / tools/list / tools/call, which are the same in all
/// three. Per the MCP lifecycle spec the server answers with the requested
/// revision when it supports it, otherwise with its newest one (the client
/// then decides whether to continue); it never confirms a revision it does
/// not implement.
constexpr std::array<std::string_view, 3> kProtocolVersions{"2025-06-18", "2025-03-26", "2024-11-05"};

std::string negotiateProtocolVersion(const json& params) {
    const auto it = params.find("protocolVersion");
    if (it != params.end() && it->is_string()) {
        const auto& asked = it->get_ref<const std::string&>();
        if (std::find(kProtocolVersions.begin(), kProtocolVersions.end(), asked) != kProtocolVersions.end()) {
            return asked;
        }
    }
    return std::string{kProtocolVersions.front()};
}

json rpcError(const json& id, int code, const std::string& message) {
    return {{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", message}}}};
}

json rpcResult(const json& id, json result) {
    return {{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}};
}

} // namespace

McpServer::McpServer(std::string name, std::string version)
    : _name(std::move(name)), _version(std::move(version)) {}

void McpServer::addTool(const std::string& name, const std::string& description,
                        nlohmann::json inputSchema, Handler handler) {
    _tools[name] = Tool{description, std::move(inputSchema), std::move(handler)};
}

json McpServer::callTool(const json& params) {
    const std::string name = params.value("name", std::string{});
    const auto it = _tools.find(name);
    if (it == _tools.end()) {
        throw std::invalid_argument("unknown tool '" + name + "'");
    }
    const json args = params.contains("arguments") ? params["arguments"] : json::object();
    try {
        json out = it->second.handler(args);
        return {{"content", json::array({{{"type", "text"}, {"text", out.dump(2)}}})},
                {"structuredContent", out},
                {"isError", false}};
    } catch (const std::exception& e) {
        return {{"content", json::array({{{"type", "text"}, {"text", std::string{e.what()}}}})},
                {"isError", true}};
    }
}

std::optional<json> McpServer::handle(const json& msg) {
    if (!msg.is_object() || msg.value("jsonrpc", std::string{}) != "2.0" ||
        !msg.contains("method") || !msg["method"].is_string()) {
        return rpcError(msg.is_object() && msg.contains("id") ? msg["id"] : json(nullptr),
                        -32600, "invalid JSON-RPC 2.0 request");
    }
    const std::string method = msg["method"];
    if (!msg.contains("id")) {
        return std::nullopt;   // notification (e.g. notifications/initialized)
    }
    const json& id     = msg["id"];
    const json  params = msg.contains("params") ? msg["params"] : json::object();

    if (method == "initialize") {
        return rpcResult(id, {{"protocolVersion", negotiateProtocolVersion(params)},
                              {"capabilities", {{"tools", json::object()}}},
                              {"serverInfo", {{"name", _name}, {"version", _version}}}});
    }
    if (method == "ping") {
        return rpcResult(id, json::object());
    }
    if (method == "tools/list") {
        json tools = json::array();
        for (const auto& [name, tool] : _tools) {
            tools.push_back({{"name", name},
                             {"description", tool.description},
                             {"inputSchema", tool.inputSchema}});
        }
        return rpcResult(id, {{"tools", std::move(tools)}});
    }
    if (method == "tools/call") {
        try {
            return rpcResult(id, callTool(params));
        } catch (const std::invalid_argument& e) {
            return rpcError(id, -32602, e.what());
        }
    }
    return rpcError(id, -32601, "method not found: " + method);
}

std::optional<std::string> McpServer::handleLine(const std::string& line) {
    json msg;
    try {
        msg = json::parse(line);
    } catch (const json::exception& e) {
        return rpcError(nullptr, -32700, std::string{"parse error: "} + e.what()).dump();
    }
    auto resp = handle(msg);
    if (!resp) {
        return std::nullopt;
    }
    return resp->dump();
}

void McpServer::run(std::istream& in, std::ostream& out) {
    std::string line;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        if (auto resp = handleLine(line)) {
            out << *resp << '\n';
            out.flush();
        }
    }
}

} // namespace mimirmind::hugin
