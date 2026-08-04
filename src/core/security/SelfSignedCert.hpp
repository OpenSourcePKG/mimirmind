// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <string>
#include <vector>

namespace mimirmind::core::security {

/// Inputs for a self-signed leaf certificate. All fields have working
/// defaults so a bare call still produces a usable localhost cert.
struct SelfSignedCertParams {
    /// Subject/issuer common name. Defaults to the resolved hostname.
    std::string              commonName{"localhost"};
    /// DNS SANs. `commonName` and `localhost` are added automatically if
    /// absent, so this only needs extra names.
    std::vector<std::string> dnsNames{};
    /// IP SANs. `127.0.0.1` is added automatically if absent.
    std::vector<std::string> ipAddresses{};
    /// Validity window in days from now.
    int                      validityDays{365};
};

/// A generated certificate + private key, both PEM-encoded. `ok == false`
/// means generation failed and `error` explains why; the PEM fields are
/// then empty.
struct SelfSignedCert {
    bool        ok{false};
    std::string error{};
    std::string certPem{};
    std::string keyPem{};
};

/**
 * Mint a self-signed X.509v3 certificate with an EC P-256 key.
 *
 * The certificate carries a random serial, a `notBefore` of now, a
 * `notAfter` `validityDays` out, `basicConstraints=CA:FALSE`, and a
 * `subjectAltName` covering the requested DNS names and IPs (modern TLS
 * clients ignore the CN and require a SAN match). It is signed with
 * SHA-256. Nothing is written to disk — the caller decides whether to
 * persist the returned PEM. All OpenSSL usage is contained in the .cpp.
 */
[[nodiscard]] SelfSignedCert generateSelfSignedCert(
    const SelfSignedCertParams& params);

/// Best-effort local hostname (`gethostname`), or "localhost" on failure.
[[nodiscard]] std::string resolveHostname();

} // namespace mimirmind::core::security
