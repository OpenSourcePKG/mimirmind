// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/security/ApiKeyStore.hpp"

#include "core/security/ConstantTimeCompare.hpp"
#include "core/log/Log.hpp"

#include <openssl/rand.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include <sys/stat.h>

namespace mimirmind::core::security {

namespace {

/// base64url alphabet (RFC 4648 §5) — URL/filename safe, no padding.
constexpr char kB64Url[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

std::string base64UrlNoPad(const unsigned char* data, std::size_t len) {
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        const std::uint32_t n = (static_cast<std::uint32_t>(data[i]) << 16) |
                                (static_cast<std::uint32_t>(data[i + 1]) << 8) |
                                static_cast<std::uint32_t>(data[i + 2]);
        out.push_back(kB64Url[(n >> 18) & 0x3F]);
        out.push_back(kB64Url[(n >> 12) & 0x3F]);
        out.push_back(kB64Url[(n >> 6) & 0x3F]);
        out.push_back(kB64Url[n & 0x3F]);
    }
    const std::size_t rem = len - i;
    if (rem == 1) {
        const std::uint32_t n = static_cast<std::uint32_t>(data[i]) << 16;
        out.push_back(kB64Url[(n >> 18) & 0x3F]);
        out.push_back(kB64Url[(n >> 12) & 0x3F]);
    } else if (rem == 2) {
        const std::uint32_t n = (static_cast<std::uint32_t>(data[i]) << 16) |
                                (static_cast<std::uint32_t>(data[i + 1]) << 8);
        out.push_back(kB64Url[(n >> 18) & 0x3F]);
        out.push_back(kB64Url[(n >> 12) & 0x3F]);
        out.push_back(kB64Url[(n >> 6) & 0x3F]);
    }
    return out;
}

} // namespace

const ApiKey* ApiKeyStore::match(std::string_view token) const {
    if (token.empty()) return nullptr;
    const ApiKey* hit = nullptr;
    // Walk every entry (no early return on match) so lookup time does not
    // reveal which slot — or whether any slot — matched. constantTimeEquals
    // guards each comparison.
    for (const ApiKey& k : _keys) {
        if (constantTimeEquals(token, k.key)) {
            hit = &k;
        }
    }
    return hit;
}

std::string ApiKeyStore::generateKey() {
    std::array<unsigned char, 32> raw{};
    if (RAND_bytes(raw.data(), static_cast<int>(raw.size())) != 1) {
        MM_LOG_ERROR("server",
                     "ApiKeyStore: RAND_bytes failed — cannot generate key");
        return {};
    }
    return "sk-mimir-" + base64UrlNoPad(raw.data(), raw.size());
}

std::vector<ApiKey> ApiKeyStore::loadKeyFile(const std::string& path) {
    std::vector<ApiKey> keys;
    std::ifstream in{path};
    if (!in) return keys;
    std::string line;
    while (std::getline(in, line)) {
        // Trim trailing CR (CRLF keyfiles) and skip blanks / comments.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const auto c1 = line.find(':');
        if (c1 == std::string::npos) continue;
        ApiKey k;
        k.name        = line.substr(0, c1);
        const auto c2 = line.find(':', c1 + 1);
        if (c2 == std::string::npos) {
            k.key      = line.substr(c1 + 1);
            k.tenantId = k.name;
        } else {
            k.key         = line.substr(c1 + 1, c2 - c1 - 1);
            // Optional 4th `:role` field after tenantId. `admin` => isAdmin.
            const auto c3 = line.find(':', c2 + 1);
            if (c3 == std::string::npos) {
                k.tenantId = line.substr(c2 + 1);
            } else {
                k.tenantId = line.substr(c2 + 1, c3 - c2 - 1);
                k.isAdmin  = (line.substr(c3 + 1) == "admin");
            }
        }
        if (k.tenantId.empty()) k.tenantId = k.name;
        if (!k.key.empty()) keys.push_back(std::move(k));
    }
    return keys;
}

bool ApiKeyStore::saveKeyFile(const std::string&         path,
                              const std::vector<ApiKey>& keys) {
    std::ostringstream body;
    body << "# mimirmind API keys — format: name:key:tenantId[:admin]\n";
    body << "# Auto-generated when server.auth is enabled and no key was "
            "configured.\n";
    for (const ApiKey& k : keys) {
        body << k.name << ':' << k.key << ':' << k.tenantId;
        if (k.isAdmin) body << ":admin";
        body << '\n';
    }
    {
        std::ofstream out{path, std::ios::binary | std::ios::trunc};
        if (!out) return false;
        out << body.str();
        if (!out) return false;
    }
    // Lock the secret down to owner-only. Best-effort: a failure here does
    // not invalidate the write, but log it so an operator can fix perms.
    if (::chmod(path.c_str(), S_IRUSR | S_IWUSR) != 0) {
        MM_LOG_WARN("server",
                    "ApiKeyStore: could not chmod 0600 keyfile {}", path);
    }
    return true;
}

} // namespace mimirmind::core::security
