// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/NvFp4Materializer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
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
                       MaterializerDeviceOps&   ops,
                       const std::function<bool(const std::string&)>&
                           deferToNvfp4Bank) {
    std::vector<MaterializedTensor> out;
    out.reserve(steps.size());

    for (const mo::MaterializationStep& step : steps) {
        // 5.27 I-2: a deferred step (routed expert repacked to an NVFP4 bank
        // later, reading the NVFP4 source directly) emits a placeholder with
        // correct metadata but NO buffer — so its BF16 image is never
        // allocated (removes the ~225 GiB transient BF16 peak for qwen4_exp).
        if (deferToNvfp4Bank && deferToNvfp4Bank(step.ggufName)) {
            out.push_back(MaterializedTensor{step.ggufName, compute::ComputeBuffer{},
                                             step.ggufDims, step.totalElems,
                                             /*isF32=*/false});
            continue;
        }
        // Passthrough tensors materialise to F32 (4 bytes), dequantised
        // NVFP4/FP8 matmul weights to BF16 (2 bytes).
        const std::size_t elemBytes = step.outF32 ? 4 : 2;
        const std::size_t outBytes  = static_cast<std::size_t>(step.totalElems) * elemBytes;
        compute::ComputeBuffer buf = ops.allocate(outBytes);
        auto* dstBase = static_cast<std::uint8_t*>(buf.get());

        for (const mo::MaterializationSource& s : step.sources) {
            void* dst = dstBase + s.dstElemOffset * elemBytes;
            // 5.27 I-2 DIAG (MIMIRMIND_Q4E_DIAG + CUDA_LAUNCH_BLOCKING): the
            // last line printed before a materialization-kernel OOB names the
            // culprit source. Temporary; remove after the qwen4_exp fix.
            static const bool kQ4eDiag = std::getenv("MIMIRMIND_Q4E_DIAG") != nullptr;
            if (kQ4eDiag) {
                std::fprintf(stderr,
                    "[q4ediag-mat] gguf=%s src=%s kind=%d rows=%llu in=%llu "
                    "dstElemOff=%llu totalElems=%llu outF32=%d\n",
                    step.ggufName.c_str(), s.hfWeightName.c_str(),
                    static_cast<int>(s.kind),
                    static_cast<unsigned long long>(s.rows),
                    static_cast<unsigned long long>(s.in),
                    static_cast<unsigned long long>(s.dstElemOffset),
                    static_cast<unsigned long long>(step.totalElems),
                    step.outF32 ? 1 : 0);
                std::fflush(stderr);
            }
            const NvFp4DeviceTensor& w = require(src, s.hfWeightName);
            const std::string base = moduleBase(s.hfWeightName);

            switch (s.kind) {
                case mo::SourceKind::Nvfp4: {
                    // ModelOpt (qwen35moe) reconstructs the scale names from the
                    // weight base; compressed-tensors (Gemma-4) supplies them
                    // explicitly (`.weight_scale` / `.weight_global_scale`, the
                    // packed weight itself being `.weight_packed`).
                    const std::string bsName = s.blockScaleName.empty()
                        ? (base + ".weight_scale") : s.blockScaleName;
                    const std::string gsName = s.globalScaleName.empty()
                        ? (base + ".weight_scale_2") : s.globalScaleName;
                    const NvFp4DeviceTensor& bs = require(src, bsName);
                    const NvFp4DeviceTensor& gs = require(src, gsName);
                    float global = ops.readF32(gs.devPtr);
                    // compressed-tensors stores 1/global (6*448/amax); the
                    // dequant kernel multiplies, so invert it here.
                    if (s.globalIsReciprocal) global = 1.0F / global;
                    ops.dequantNvfp4(w.devPtr, bs.devPtr, global, s.rows, s.in, dst);
                    break;
                }
                case mo::SourceKind::Fp8: {
                    const NvFp4DeviceTensor& ws = require(src, base + ".weight_scale");
                    const float scale = ops.readF32(ws.devPtr);
                    ops.dequantFp8(w.devPtr, scale, s.rows * s.in, dst);
                    break;
                }
                case mo::SourceKind::Fp8PerChannel: {
                    // compressed-tensors FP8: `<base>.weight_scale` is a
                    // per-output-row BF16 vector [rows], not a per-tensor F32
                    // scalar. Pass the whole scale vector to the kernel.
                    const NvFp4DeviceTensor& ws = require(src, base + ".weight_scale");
                    if (ws.dtype != safetensors::SafetensorsDtype::BF16) {
                        fail("per-channel FP8 scale '" + base +
                             ".weight_scale' is not BF16 (the kernel widens BF16 "
                             "per-channel scales); got dtype "
                             + std::string{safetensors::dtypeName(ws.dtype)});
                    }
                    ops.dequantFp8PerChannel(w.devPtr, ws.devPtr, s.rows, s.in, dst);
                    break;
                }
                case mo::SourceKind::Bf16Copy: {
                    // Already-BF16 matmul weight, kept BF16 (no widen). Reads a
                    // slice at `srcElemOffset` — used to split the MTP fused
                    // gate_up_proj and de-stack the expert-major MTP MoE tensors
                    // into per-expert GGUF ffn_*_exps. 2 bytes/element.
                    const auto* srcBytes =
                        static_cast<const std::uint8_t*>(w.devPtr) + s.srcElemOffset * 2;
                    ops.copyBytes(dst, srcBytes, s.rows * s.in * 2);
                    break;
                }
                case mo::SourceKind::Bf16Passthrough:
                default:
                    // Unquantised: widen BF16/F16 -> F32 (or copy F32) so the
                    // runtime can read it as a `const float*`.
                    ops.widenToF32(dst, w.devPtr, w.dtype, s.rows * s.in);
                    break;
            }
        }

        // Element-wise fix-up on the finished buffer, only on F32 passthrough
        // outputs. NegExp: HF `A_log` -> GGUF `ssm_a` (= -exp(A_log)) for the
        // DeltaNet decay gate. AddOne: HF centred RMSNorm weight -> GGUF
        // (1 + w) for the transformer norms.
        if (step.postTransform != mo::PostTransform::None && !step.outF32) {
            fail("post-transform requires an F32 output ('" + step.ggufName + "')");
        }
        if (step.postTransform == mo::PostTransform::NegExp) {
            ops.negExpInPlaceF32(dstBase, step.totalElems);
        } else if (step.postTransform == mo::PostTransform::AddOne) {
            ops.addOneInPlaceF32(dstBase, step.totalElems);
        }

        out.push_back(MaterializedTensor{step.ggufName, std::move(buf),
                                         step.ggufDims, step.totalElems,
                                         step.outF32});
    }

    return out;
}

} // namespace mimirmind::runtime::nvfp4