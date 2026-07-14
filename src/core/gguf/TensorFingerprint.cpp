// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gguf/TensorFingerprint.hpp"

#include "core/gguf/GgufReader.hpp"

#include <cstdint>
#include <iomanip>
#include <sstream>

namespace mimirmind::core::gguf {

namespace {

constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnvPrime  = 0x100000001b3ULL;

inline void fnvUpdate(std::uint64_t& h, const void* data, std::size_t n) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= kFnvPrime;
    }
}

std::string toHex64(std::uint64_t v) {
    std::ostringstream os;
    os << std::hex << std::setw(16) << std::setfill('0') << v;
    return os.str();
}

} // namespace

std::string tensorFingerprint(const GgufReader& reader) {
    std::uint64_t h = kFnvOffset;

    const std::uint32_t v = reader.version();
    fnvUpdate(h, &v, sizeof(v));
    const std::uint64_t a = reader.alignment();
    fnvUpdate(h, &a, sizeof(a));
    const std::uint64_t off = reader.tensorDataOffset();
    fnvUpdate(h, &off, sizeof(off));
    const std::uint64_t total = reader.totalTensorBytes();
    fnvUpdate(h, &total, sizeof(total));

    const std::uint64_t mdCount = reader.metadataCount();
    fnvUpdate(h, &mdCount, sizeof(mdCount));

    const auto& ts = reader.tensors();
    const std::uint64_t nT = ts.size();
    fnvUpdate(h, &nT, sizeof(nT));
    for (const auto& t : ts) {
        fnvUpdate(h, t.name.data(), t.name.size());
        const std::uint32_t ty = static_cast<std::uint32_t>(t.type);
        fnvUpdate(h, &ty, sizeof(ty));
        const std::uint64_t nd = t.dimensions.size();
        fnvUpdate(h, &nd, sizeof(nd));
        if (!t.dimensions.empty()) {
            fnvUpdate(h, t.dimensions.data(),
                      t.dimensions.size() * sizeof(std::uint64_t));
        }
        const std::uint64_t nb = t.nbytes;
        fnvUpdate(h, &nb, sizeof(nb));
        fnvUpdate(h, &t.fileOffset, sizeof(t.fileOffset));
    }

    std::ostringstream os;
    os << ts.size() << "." << reader.totalTensorBytes() << "." << toHex64(h);
    return os.str();
}

} // namespace mimirmind::core::gguf