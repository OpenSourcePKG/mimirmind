// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/rune/Futhark.hpp"

#include "core/gguf/GgufReader.hpp"
#include "core/gguf/WeightsMap.hpp"

#include <map>
#include <stdexcept>

namespace mimirmind::core::rune {

namespace {

RuneElemType elemTypeOf(gguf::GgmlType t) noexcept {
    switch (t) {
        case gguf::GgmlType::F32:  return RuneElemType::F32;
        case gguf::GgmlType::F16:  return RuneElemType::F16;
        case gguf::GgmlType::BF16: return RuneElemType::BF16;
        default:                   return RuneElemType::Unsupported;
    }
}

/// Host staging copy of one device tensor.
struct Staged {
    const gguf::GgufTensor* tensor{nullptr};
    std::vector<std::byte>  host;
};

} // namespace

bool Futhark::hasDerivedCopies(const gguf::GgufTensor& t, const gguf::WeightsMap& weights) {
    if (t.tcNibblePtr != nullptr || t.tcSfbPtr != nullptr || t.tcGlobalsPtr != nullptr) {
        return true;
    }
    for (const auto suffix : kDerivedSuffixes) {
        if (weights.find(t.name + std::string{suffix}) != nullptr) {
            return true;
        }
    }
    return false;
}

std::expected<Futhark::Result, std::string>
Futhark::applyFiles(std::span<const std::string> runePaths, std::string_view baseFingerprint,
                    const gguf::WeightsMap& weights) {
    std::vector<RuneFile> runes;
    runes.reserve(runePaths.size());
    for (const auto& path : runePaths) {
        try {
            runes.push_back(RuneFile::load(path));
        } catch (const std::exception& e) {
            return std::unexpected(std::string{e.what()});
        }
    }
    return apply(runes, baseFingerprint, weights);
}

std::expected<Futhark::Result, std::string>
Futhark::apply(std::span<const RuneFile> runes, std::string_view baseFingerprint,
               const gguf::WeightsMap& weights) {
    // Keyed by name; std::map keeps node addresses stable, so the spans handed
    // to the applier stay valid while later runes stage more tensors.
    std::map<std::string, Staged, std::less<>> staged;
    std::string                                ioError;

    const RuneTensorResolver resolve =
        [&](std::string_view name) -> std::optional<RuneTensorView> {
        const gguf::GgufTensor* t = weights.find(name);
        if (t == nullptr || t->usmPtr == nullptr) {
            return std::nullopt;
        }
        RuneTensorView v;
        v.name             = t->name;
        v.type             = elemTypeOf(t->type);
        v.dims             = t->dimensions;
        v.hasDerivedCopies = hasDerivedCopies(*t, weights);
        if (v.type == RuneElemType::Unsupported || v.hasDerivedCopies) {
            return v;   // the applier rejects it with a precise reason; no readback
        }
        auto it = staged.find(name);
        if (it == staged.end()) {
            Staged s{t, std::vector<std::byte>(t->nbytes)};
            try {
                _io.readback(s.host.data(), t->usmPtr, t->nbytes);
            } catch (const std::exception& e) {
                ioError = "readback of '" + t->name + "' failed: " + e.what();
                return std::nullopt;
            }
            it = staged.emplace(t->name, std::move(s)).first;
        }
        v.bytes = std::span<std::byte>(it->second.host);
        return v;
    };

    Result      result;
    RuneApplier applier;   // host backups only; discarded with the staging
    for (const auto& rune : runes) {
        auto rep = applier.apply(rune, baseFingerprint, resolve);
        if (!rep) {
            return std::unexpected(ioError.empty() ? rep.error() : ioError);
        }
        result.reports.push_back(std::move(*rep));
    }

    // Every rune applied on staging — now (and only now) touch the device.
    std::vector<const Staged*> uploaded;
    for (const auto& report : result.reports) {
        for (const auto& touched : report.touched) {
            const auto it = staged.find(touched.name);
            if (it == staged.end()) {
                continue;
            }
            const Staged& s = it->second;
            bool already = false;   // a tensor touched by several runes uploads once
            for (const Staged* u : uploaded) {
                already = already || u == &s;
            }
            if (already) {
                continue;
            }
            try {
                _io.upload(s.tensor->usmPtr, s.host.data(), s.host.size());
            } catch (const std::exception& e) {
                // Restore staging to the base bytes and push them back over what
                // was already uploaded, so the device ends up untouched.
                // The failed tensor is included: its copy may have landed partially.
                applier.rollback();
                uploaded.push_back(&s);
                std::string restoreNote = "device weights restored to the base model";
                for (const Staged* u : uploaded) {
                    try {
                        _io.upload(u->tensor->usmPtr, u->host.data(), u->host.size());
                    } catch (const std::exception&) {
                        restoreNote = "restore upload ALSO failed — reload the model";
                    }
                }
                return std::unexpected("upload of '" + s.tensor->name + "' failed: " +
                                       e.what() + "; " + restoreNote);
            }
            uploaded.push_back(&s);
            ++result.tensorsUploaded;
            result.bytesUploaded += s.host.size();
        }
    }
    return result;
}

} // namespace mimirmind::core::rune
