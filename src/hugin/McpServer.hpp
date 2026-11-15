// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <nlohmann/json.hpp>

#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>

namespace mimirmind::hugin {

/**
 * Minimal Model Context Protocol server over stdio (newline-delimited
 * JSON-RPC 2.0), the transport Claude Code uses for local MCP servers.
 *
 * Implements `initialize`, `notifications/initialized`, `ping`, `tools/list`
 * and `tools/call`. A tool handler receives the call's `arguments` object and
 * returns a JSON result; a thrown exception becomes a tool result with
 * `isError: true` (a tool failure, not a protocol error), so Claude sees the
 * message and can react.
 */
class McpServer {
public:
    using Handler = std::function<nlohmann::json(const nlohmann::json& args)>;

    McpServer(std::string name, std::string version);

    /// Register a tool. `inputSchema` is a JSON Schema object.
    void addTool(const std::string& name, const std::string& description,
                 nlohmann::json inputSchema, Handler handler);

    /// Handle one JSON-RPC message. Returns the response, or nullopt for
    /// notifications (no `id`).
    [[nodiscard]] std::optional<nlohmann::json> handle(const nlohmann::json& msg);

    /// Handle one raw line (parse errors produce a JSON-RPC -32700 response).
    [[nodiscard]] std::optional<std::string> handleLine(const std::string& line);

    /// Serve until EOF on `in`. Each response is one line on `out`, flushed.
    void run(std::istream& in, std::ostream& out);

private:
    struct Tool {
        std::string    description;
        nlohmann::json inputSchema;
        Handler        handler;
    };

    [[nodiscard]] nlohmann::json callTool(const nlohmann::json& params);

    std::string                 _name;
    std::string                 _version;
    std::map<std::string, Tool> _tools;
};

} // namespace mimirmind::hugin
