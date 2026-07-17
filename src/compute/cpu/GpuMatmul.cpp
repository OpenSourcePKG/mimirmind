// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/cpu/GpuMatmul.hpp"

#include "compute/Matmul.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "core/cpu/CpuContext.hpp"

#include <stdexcept>
#include <string>

namespace mimirmind::compute::cpu {

namespace {

[[noreturn]] void throwNotImplemented(const char* method) {
    throw std::runtime_error(
        std::string{"compute::cpu::GpuMatmul::"} + method +
        ": not implemented on the CPU backend — no reference kernel "
        "for this fusion path (see M-CPU-Backend milestone in "
        "Synaipse todos/m-cpu-backend for scope)");
}

} // namespace

GpuMatmul::GpuMatmul(::mimirmind::core::cpu::CpuContext& ctx)
    : _ctx{ctx} {}

bool GpuMatmul::supports(::mimirmind::core::gguf::GgmlType type)
    const noexcept {
    // Every type with a QuantTypeRegistry entry is dispatchable — the
    // reference `compute::matmul` walks the row dequant through
    // dequantToF32 which resolves through the same registry. Types
    // without an entry (Q4_0 / Q2_K / Q3_K / ...) fall through to false.
    return ::mimirmind::compute::quantType(type) != nullptr;
}

void GpuMatmul::matmul(::mimirmind::core::gguf::GgmlType type,
                       const void*  W,
                       std::size_t  N,
                       std::size_t  K,
                       const float* X,
                       std::size_t  M,
                       float*       Y,
                       float*       scratch) {
    ::mimirmind::compute::matmul(type, W, N, K, X, M, Y, scratch);
}

void GpuMatmul::matmulAsync(::mimirmind::core::gguf::GgmlType type,
                            const void*  W,
                            std::size_t  N,
                            std::size_t  K,
                            const float* X,
                            std::size_t  M,
                            float*       Y,
                            float*       scratch) {
    // CPU is synchronous — the "async" method returns after the compute
    // is complete. Matches how L0 GpuMatmul falls back to
    // `compute::matmul` for types without a GPU kernel: the semantics
    // are simply "run it inline".
    ::mimirmind::compute::matmul(type, W, N, K, X, M, Y, scratch);
}

void GpuMatmul::matmulDp4aAsync(::mimirmind::core::gguf::GgmlType /*type*/,
                                const std::int8_t* /*Xq*/,
                                const float*       /*Xscale*/,
                                const void*        /*W*/,
                                std::size_t        /*N*/,
                                std::size_t        /*K*/,
                                std::size_t        /*M*/,
                                float*             /*Y*/) {
    // Callers should have gated on dp4aAvailable() first — it returns
    // false unconditionally. Throw so a bypass call is loud rather
    // than silently producing zeros.
    throwNotImplemented("matmulDp4aAsync");
}

void GpuMatmul::moeDownFusedKAsync(::mimirmind::core::gguf::GgmlType /*type*/,
                                   const float*         /*gateAct*/,
                                   const void*          /*W*/,
                                   const std::int32_t*  /*expIdx*/,
                                   const float*         /*kw*/,
                                   float*               /*accum*/,
                                   std::size_t          /*ffPer*/,
                                   std::size_t          /*dModel*/,
                                   std::size_t          /*kActive*/,
                                   std::size_t          /*expertBytes*/) {
    // Same as DP4A above — moeDownFusedKAvailable() returns false, so
    // dispatch here is a bug. Loud throw over silent corruption.
    throwNotImplemented("moeDownFusedKAsync");
}

} // namespace mimirmind::compute::cpu