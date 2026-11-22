// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/SharedArenaWeights.hpp"

#include "core/ipc/SharedWeightArena.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace mimirmind::runtime::nvfp4 {

namespace {

using core::ipc::SharedWeightArena;

struct FlagBinding {
    const char*            name;
    bool MaterializedTensor::*field;
};

constexpr FlagBinding kFlags[] = {
    {"f32", &MaterializedTensor::isF32},
    {"q8_0", &MaterializedTensor::isQ8_0},
    {"q4k", &MaterializedTensor::isQ4K},
    {"q6k", &MaterializedTensor::isQ6K},
    {"fp8", &MaterializedTensor::isFp8},
    {"nvfp4blk", &MaterializedTensor::isNvfp4Blk},
    {"nvfp4tc", &MaterializedTensor::isNvfp4Tc},
};

struct BufferBinding {
    const char*                                 role;
    compute::ComputeBuffer MaterializedTensor::*field;
};

constexpr BufferBinding kBuffers[] = {
    {"data", &MaterializedTensor::buffer},
    {"tcNibble", &MaterializedTensor::tcNibbleBank},
    {"tcSfb", &MaterializedTensor::tcSfbBank},
    {"tcGlobals", &MaterializedTensor::tcGlobalsBank},
};

void arenaFree(void* p, std::size_t, void* ctx) noexcept {
    static_cast<SharedWeightArena*>(ctx)->deallocate(p);
}

} // namespace

std::size_t SharedArenaWeights::migrateInto(std::vector<MaterializedTensor>& mats,
                                            SharedWeightArena&               arena,
                                            const CopyFn&                    copy) {
    std::size_t moved = 0;
    for (auto& t : mats) {
        for (const auto& bb : kBuffers) {
            compute::ComputeBuffer& buf = t.*(bb.field);
            if (buf.get() == nullptr || buf.bytes() == 0 || arena.contains(buf.get())) {
                continue;
            }
            void* dst = arena.allocate(buf.bytes());
            if (dst == nullptr) {
                throw std::runtime_error("shared weight arena exhausted while moving '" +
                                         t.ggufName + "' (" + bb.role + ")");
            }
            copy(dst, buf.get(), buf.bytes());
            buf = compute::ComputeBuffer{dst, buf.bytes(), &arenaFree, &arena};
            ++moved;
        }
    }
    return moved;
}

core::ipc::WeightArenaManifest
SharedArenaWeights::describe(const std::vector<MaterializedTensor>& mats,
                             const SharedWeightArena&               arena) {
    core::ipc::WeightArenaManifest m;
    std::uint64_t extent = 0;
    for (const auto& t : mats) {
        core::ipc::WeightArenaTensor wt;
        wt.name  = t.ggufName;
        wt.dims  = t.ggufDims;
        wt.elems = t.elems;
        for (const auto& f : kFlags) {
            if (t.*(f.field)) {
                wt.flags.emplace_back(f.name);
            }
        }
        for (const auto& bb : kBuffers) {
            const compute::ComputeBuffer& buf = t.*(bb.field);
            if (buf.get() == nullptr || buf.bytes() == 0) {
                continue;
            }
            if (!arena.contains(buf.get())) {
                throw std::runtime_error("shared weight arena: tensor '" + t.ggufName +
                                         "' buffer '" + bb.role +
                                         "' lies outside the arena");
            }
            const std::uint64_t off = arena.offsetOf(buf.get());
            wt.buffers.push_back({bb.role, off, buf.bytes()});
            extent = std::max<std::uint64_t>(extent, off + buf.bytes());
        }
        m.tensors.push_back(std::move(wt));
    }
    m.mapBytes = (extent + SharedWeightArena::kPageBytes - 1) /
                 SharedWeightArena::kPageBytes * SharedWeightArena::kPageBytes;
    return m;
}

std::vector<MaterializedTensor>
SharedArenaWeights::bind(const core::ipc::WeightArenaManifest& manifest, void* base) {
    std::vector<MaterializedTensor> out;
    out.reserve(manifest.tensors.size());
    auto* b = static_cast<std::uint8_t*>(base);
    for (const auto& wt : manifest.tensors) {
        MaterializedTensor t;
        t.ggufName = wt.name;
        t.ggufDims = wt.dims;
        t.elems    = wt.elems;
        for (const auto& name : wt.flags) {
            const auto* f = std::find_if(std::begin(kFlags), std::end(kFlags),
                                         [&](const FlagBinding& x) { return name == x.name; });
            if (f == std::end(kFlags)) {
                throw std::runtime_error("shared weight arena: tensor '" + wt.name +
                                         "' has unknown format flag '" + name +
                                         "' (engine/attacher version skew)");
            }
            t.*(f->field) = true;
        }
        for (const auto& wb : wt.buffers) {
            const auto* bb = std::find_if(std::begin(kBuffers), std::end(kBuffers),
                                          [&](const BufferBinding& x) { return wb.role == x.role; });
            if (bb == std::end(kBuffers)) {
                throw std::runtime_error("shared weight arena: tensor '" + wt.name +
                                         "' has unknown buffer role '" + wb.role + "'");
            }
            if (wb.offset + wb.bytes > manifest.mapBytes) {
                throw std::runtime_error("shared weight arena: tensor '" + wt.name +
                                         "' lies outside the mapped extent");
            }
            // Non-owning view: no deleter, the attached mapping owns the bytes.
            t.*(bb->field) = compute::ComputeBuffer{b + wb.offset, wb.bytes, nullptr, nullptr};
        }
        out.push_back(std::move(t));
    }
    return out;
}

} // namespace mimirmind::runtime::nvfp4
