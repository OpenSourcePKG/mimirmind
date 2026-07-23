// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/NvFp4Materializer.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace mimirmind::runtime::nvfp4 {

namespace mo = core::modelopt;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("nvfp4 materialize: " + msg);
}

const NvFp4DeviceTensor& require(const NvFp4Model& src, const std::string& name) {
    const NvFp4DeviceTensor* t = src.find(name);
    if (t == nullptr) {
        fail("missing device tensor '" + name + "'");
    }
    return *t;
}

/// Strip a trailing ".weight" to get the module base for scale sidecars.
std::string moduleBase(const std::string& weightName) {
    static const std::string suf = ".weight";
    if (weightName.size() >= suf.size()
        && weightName.compare(weightName.size() - suf.size(), suf.size(), suf) == 0) {
        return weightName.substr(0, weightName.size() - suf.size());
    }
    return weightName;
}

} // namespace

std::vector<MaterializedTensor>
executeMaterialization(const std::vector<mo::MaterializationStep>& steps,
                       const NvFp4Model&        src,
                       MaterializerDeviceOps&   ops) {
    std::vector<MaterializedTensor> out;
    out.reserve(steps.size());

    for (const mo::MaterializationStep& step : steps) {
        const std::size_t bf16Bytes = static_cast<std::size_t>(step.totalElems) * 2;
        compute::ComputeBuffer buf = ops.allocate(bf16Bytes);
        auto* dstBase = static_cast<std::uint8_t*>(buf.get());

        for (const mo::MaterializationSource& s : step.sources) {
            void* dst = dstBase + s.dstElemOffset * 2; // BF16 = 2 bytes/element
            const NvFp4DeviceTensor& w = require(src, s.hfWeightName);
            const std::string base = moduleBase(s.hfWeightName);

            switch (s.kind) {
                case mo::SourceKind::Nvfp4: {
                    const NvFp4DeviceTensor& bs = require(src, base + ".weight_scale");
                    const NvFp4DeviceTensor& gs = require(src, base + ".weight_scale_2");
                    const float global = ops.readF32(gs.devPtr);
                    ops.dequantNvfp4(w.devPtr, bs.devPtr, global, s.rows, s.in, dst);
                    break;
                }
                case mo::SourceKind::Fp8: {
                    const NvFp4DeviceTensor& ws = require(src, base + ".weight_scale");
                    const float scale = ops.readF32(ws.devPtr);
                    ops.dequantFp8(w.devPtr, scale, s.rows * s.in, dst);
                    break;
                }
                case mo::SourceKind::Bf16Passthrough:
                default:
                    // Already BF16 on device; copy its bytes verbatim.
                    ops.copyBytes(dst, w.devPtr, w.nbytes);
                    break;
            }
        }

        out.push_back(MaterializedTensor{step.ggufName, std::move(buf),
                                         step.ggufDims, step.totalElems});
    }

    return out;
}

} // namespace mimirmind::runtime::nvfp4