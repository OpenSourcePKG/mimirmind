// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/security/SelfSignedCert.hpp"

#include <openssl/bio.h>
#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include <algorithm>
#include <ctime>
#include <memory>

#include <unistd.h>

namespace mimirmind::core::security {

namespace {

// RAII deleters keep the OpenSSL object graph exception- and
// early-return-safe — no manual free ladder.
struct EvpPkeyDeleter { void operator()(EVP_PKEY* p) const { EVP_PKEY_free(p); } };
struct X509Deleter    { void operator()(X509*     p) const { X509_free(p); } };
struct BioDeleter     { void operator()(BIO*      p) const { BIO_free(p); } };
struct BnDeleter      { void operator()(BIGNUM*   p) const { BN_free(p); } };

using EvpPkeyPtr = std::unique_ptr<EVP_PKEY, EvpPkeyDeleter>;
using X509Ptr    = std::unique_ptr<X509,     X509Deleter>;
using BioPtr     = std::unique_ptr<BIO,      BioDeleter>;
using BnPtr      = std::unique_ptr<BIGNUM,   BnDeleter>;

std::string opensslError() {
    unsigned long e = ERR_get_error();
    if (e == 0) return "unknown OpenSSL error";
    char buf[256];
    ERR_error_string_n(e, buf, sizeof(buf));
    return buf;
}

// Read a memory BIO into a std::string (PEM output).
std::string bioToString(BIO* bio) {
    char*      data = nullptr;
    const long len  = BIO_get_mem_data(bio, &data);
    if (data == nullptr || len <= 0) return {};
    return std::string(data, static_cast<std::size_t>(len));
}

bool addName(X509_NAME* name, const char* field, const std::string& value) {
    return X509_NAME_add_entry_by_txt(
               name, field, MBSTRING_ASC,
               reinterpret_cast<const unsigned char*>(value.c_str()),
               -1, -1, 0) == 1;
}

} // namespace

std::string resolveHostname() {
    char host[256] = {0};
    if (::gethostname(host, sizeof(host) - 1) == 0 && host[0] != '\0') {
        return std::string(host);
    }
    return "localhost";
}

SelfSignedCert generateSelfSignedCert(const SelfSignedCertParams& params) {
    SelfSignedCert result;

    // ---- EC P-256 key pair -------------------------------------------------
    EvpPkeyPtr pkey{EVP_EC_gen("P-256")};
    if (!pkey) {
        result.error = "EC P-256 key generation failed: " + opensslError();
        return result;
    }

    // ---- Certificate shell -------------------------------------------------
    X509Ptr x509{X509_new()};
    if (!x509) {
        result.error = "X509_new failed: " + opensslError();
        return result;
    }
    X509_set_version(x509.get(), 2);   // X.509 v3 (0-based)

    // Random 64-bit positive serial (avoids the reuse/collision that a
    // fixed serial causes for pinning clients).
    BnPtr serial{BN_new()};
    if (!serial || BN_rand(serial.get(), 64, BN_RAND_TOP_ANY,
                           BN_RAND_BOTTOM_ANY) != 1 ||
        BN_to_ASN1_INTEGER(serial.get(),
                           X509_get_serialNumber(x509.get())) == nullptr) {
        result.error = "serial generation failed: " + opensslError();
        return result;
    }

    // Validity window.
    const long span = static_cast<long>(params.validityDays) * 24 * 60 * 60;
    if (X509_gmtime_adj(X509_getm_notBefore(x509.get()), 0) == nullptr ||
        X509_gmtime_adj(X509_getm_notAfter(x509.get()), span) == nullptr) {
        result.error = "validity set failed: " + opensslError();
        return result;
    }

    if (X509_set_pubkey(x509.get(), pkey.get()) != 1) {
        result.error = "X509_set_pubkey failed: " + opensslError();
        return result;
    }

    // Subject == issuer (self-signed). CN = requested common name.
    X509_NAME* name = X509_get_subject_name(x509.get());
    const std::string cn =
        params.commonName.empty() ? "localhost" : params.commonName;
    if (!addName(name, "CN", cn) ||
        X509_set_issuer_name(x509.get(), name) != 1) {
        result.error = "subject/issuer name set failed: " + opensslError();
        return result;
    }

    // ---- v3 extensions: basicConstraints + SAN ----------------------------
    X509V3_CTX ctx;
    X509V3_set_ctx_nodb(&ctx);
    X509V3_set_ctx(&ctx, x509.get(), x509.get(), nullptr, nullptr, 0);

    if (X509_EXTENSION* bc = X509V3_EXT_conf_nid(
            nullptr, &ctx, NID_basic_constraints, "critical,CA:FALSE")) {
        X509_add_ext(x509.get(), bc, -1);
        X509_EXTENSION_free(bc);
    }

    // Assemble the SAN value: DNS:<cn>, DNS:localhost, IP:127.0.0.1 plus
    // any caller extras, de-duplicated so clients get a clean list.
    std::vector<std::string> dns = params.dnsNames;
    if (std::find(dns.begin(), dns.end(), cn) == dns.end()) dns.push_back(cn);
    if (std::find(dns.begin(), dns.end(), "localhost") == dns.end())
        dns.push_back("localhost");
    std::vector<std::string> ips = params.ipAddresses;
    if (std::find(ips.begin(), ips.end(), "127.0.0.1") == ips.end())
        ips.push_back("127.0.0.1");

    std::string san;
    for (const std::string& d : dns) {
        if (!san.empty()) san += ",";
        san += "DNS:" + d;
    }
    for (const std::string& ip : ips) {
        if (!san.empty()) san += ",";
        san += "IP:" + ip;
    }
    if (X509_EXTENSION* ext = X509V3_EXT_conf_nid(
            nullptr, &ctx, NID_subject_alt_name, san.c_str())) {
        X509_add_ext(x509.get(), ext, -1);
        X509_EXTENSION_free(ext);
    } else {
        result.error = "subjectAltName build failed: " + opensslError();
        return result;
    }

    // ---- Self-sign (SHA-256) ----------------------------------------------
    if (X509_sign(x509.get(), pkey.get(), EVP_sha256()) == 0) {
        result.error = "X509_sign failed: " + opensslError();
        return result;
    }

    // ---- PEM-encode both ---------------------------------------------------
    BioPtr certBio{BIO_new(BIO_s_mem())};
    BioPtr keyBio{BIO_new(BIO_s_mem())};
    if (!certBio || !keyBio) {
        result.error = "BIO alloc failed: " + opensslError();
        return result;
    }
    if (PEM_write_bio_X509(certBio.get(), x509.get()) != 1) {
        result.error = "PEM_write_bio_X509 failed: " + opensslError();
        return result;
    }
    if (PEM_write_bio_PrivateKey(keyBio.get(), pkey.get(), nullptr, nullptr,
                                 0, nullptr, nullptr) != 1) {
        result.error = "PEM_write_bio_PrivateKey failed: " + opensslError();
        return result;
    }

    result.certPem = bioToString(certBio.get());
    result.keyPem  = bioToString(keyBio.get());
    if (result.certPem.empty() || result.keyPem.empty()) {
        result.error = "empty PEM output";
        return result;
    }
    result.ok = true;
    return result;
}

} // namespace mimirmind::core::security
