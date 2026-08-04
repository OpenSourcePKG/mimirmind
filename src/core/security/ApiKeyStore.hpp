// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::security {

/**
 * One bearer API key with its tenant binding.
 *
 * `key` is the raw secret sent as `Authorization: Bearer <key>`. `name`
 * is an operator-facing label (keyfile column / log line). `tenantId`
 * feeds later per-tenant serving admission — defaults to the key's `name`,
 * or `"default"` for the auto-generated key.
 */
struct ApiKey {
    std::string name{};
    std::string key{};
    std::string tenantId{};
};

/**
 * In-memory set of accepted API keys plus a constant-time lookup.
 *
 * Loaded once at startup (from explicit config `keys[]`, a keyfile, or a
 * single auto-generated key) and then only read from the request path, so
 * it is effectively immutable while the server runs — no internal locking.
 * `match()` uses a constant-time byte compare so a valid-length guess
 * cannot be recovered by timing.
 */
class ApiKeyStore {
public:
    ApiKeyStore() = default;

    void add(ApiKey key) { _keys.push_back(std::move(key)); }

    [[nodiscard]] bool        empty()  const { return _keys.empty(); }
    [[nodiscard]] std::size_t size()   const { return _keys.size(); }
    [[nodiscard]] const std::vector<ApiKey>& keys() const { return _keys; }

    /// Constant-time lookup of a presented bearer token. Returns the
    /// matching ApiKey, or nullptr when no key matches (or `token` empty).
    [[nodiscard]] const ApiKey* match(std::string_view token) const;

    // ---- Zero-config provisioning helpers ----------------------------------

    /// Draw a fresh cryptographically-random key shaped like an OpenAI
    /// secret: `sk-mimir-<43 base64url chars>` (32 random bytes). Uses the
    /// OpenSSL CSPRNG (RAND_bytes). Returns an empty string if the CSPRNG
    /// fails.
    [[nodiscard]] static std::string generateKey();

    /// Parse a keyfile. One key per non-empty, non-`#` line in the form
    /// `name:key[:tenantId]`. Missing tenantId defaults to `name`. Missing
    /// or unreadable file yields an empty vector (not an error).
    [[nodiscard]] static std::vector<ApiKey> loadKeyFile(const std::string& path);

    /// Persist keys to `path`, one `name:key:tenantId` line each, and set
    /// mode 0600. Returns false if the file could not be written (caller
    /// then falls back to keeping the key in memory only).
    [[nodiscard]] static bool saveKeyFile(const std::string&         path,
                                          const std::vector<ApiKey>& keys);

private:
    std::vector<ApiKey> _keys{};
};

} // namespace mimirmind::core::security
