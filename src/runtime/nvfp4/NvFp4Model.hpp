// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "core/modelopt/HfQuantConfig.hpp"
#include "core/modelopt/ModelOptWeightLayout.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime::nvfp4 {

namespace safetensors = mimirmind::core::safetensors;
namespace modelopt    = mimirmind::core::modelopt;

/**
 * The narrow device-upload seam the NVFP4 loader needs — allocate a device
 * buffer and blocking-copy host bytes into it. `compute::ComputeOps`
 * satisfies this shape (adapted by `ComputeOpsUploader`), and a test double
 * can too, so the loader is exercisable without dragging the whole compute
 * backend in. Mirrors the two `ComputeOps` calls `GgufReader::loadTensors`
 * uses.
 */
struct DeviceUploader {
    virtual ~DeviceUploader() = default;
    [[nodiscard]] virtual compute::ComputeBuffer allocate(std::size_t bytes) = 0;
    virtual void uploadHostBytes(void* deviceDst, const void* hostSrc,
                                 std::size_t bytes) = 0;
};

/// One device-resident tensor: where it landed and what it is. `devPtr`
/// points into a `compute::ComputeBuffer` owned by the `NvFp4Model`.
struct NvFp4DeviceTensor {
    void*                         devPtr{nullptr};
    std::size_t                   nbytes{0};
    safetensors::SafetensorsDtype dtype{safetensors::SafetensorsDtype::Unknown};
    std::vector<std::uint64_t>    shape;
};

/**
 * One assembled, device-resident ModelOpt weight: the validated logical
 * layout plus device pointers to the packed weight and its scale sidecars.
 * The populated scale members depend on the scheme (NVFP4 -> block+global,
 * FP8 -> weight+input); the rest have a null `devPtr`. This is the shape a
 * Phase-C FP4/FP8 GEMM kernel will consume.
 */
struct NvFp4DeviceWeight {
    modelopt::ModelOptWeightLayout layout;
    NvFp4DeviceTensor              packedWeight;
    NvFp4DeviceTensor              blockScale;   ///< NVFP4 per-group F8_E4M3
    NvFp4DeviceTensor              globalScale;  ///< NVFP4 per-tensor F32
    NvFp4DeviceTensor              weightScale;  ///< FP8 per-tensor F32
    NvFp4DeviceTensor              inputScale;   ///< FP8 per-tensor F32
};

/**
 * A ModelOpt NVFP4 checkpoint loaded onto a compute device. Every
 * safetensors tensor (BF16 embeddings/norms/router/MTP/vision, the packed
 * NVFP4/FP8 weights, and all scale sidecars) is uploaded and addressable by
 * name; quantised modules additionally expose an assembled + validated
 * `NvFp4DeviceWeight`.
 *
 * Owns the device allocations (move-only). This is the NVFP4 analogue of
 * `GgufReader::loadTensors` + `WeightsMap`, but tensor names are the HF
 * checkpoint names (`model.language_model.layers.N...`), NOT the GGUF /
 * llama.cpp convention — bridging those is a later (Phase C) concern.
 */
class NvFp4Model {
public:
    NvFp4Model() = default;

    NvFp4Model(const NvFp4Model&)            = delete;
    NvFp4Model& operator=(const NvFp4Model&) = delete;
    NvFp4Model(NvFp4Model&&) noexcept            = default;
    NvFp4Model& operator=(NvFp4Model&&) noexcept = default;

    /// Any uploaded tensor by exact HF name, or nullptr.
    [[nodiscard]] const NvFp4DeviceTensor* find(std::string_view name) const noexcept;

    /// Assembled weight for a quantised module base (e.g.
    /// "...mlp.experts.5.gate_proj", no `.weight` suffix), or nullptr if the
    /// module is not quantised.
    [[nodiscard]] const NvFp4DeviceWeight* weight(std::string_view module) const noexcept;

    [[nodiscard]] std::size_t   tensorCount()          const noexcept { return _tensors.size(); }
    [[nodiscard]] std::size_t   quantizedWeightCount() const noexcept { return _weights.size(); }
    [[nodiscard]] std::uint64_t deviceBytes()          const noexcept { return _deviceBytes; }
    [[nodiscard]] const modelopt::HfQuantConfig& quantConfig() const noexcept { return _config; }

private:
    friend NvFp4Model loadNvfp4Model(const std::string& checkpointDir,
                                     DeviceUploader&    uploader);

    std::vector<compute::ComputeBuffer>       _buffers;   ///< owns device memory
    std::map<std::string, NvFp4DeviceTensor>  _tensors;   ///< HF name -> device tensor
    std::map<std::string, NvFp4DeviceWeight>  _weights;   ///< module base -> assembled
    modelopt::HfQuantConfig                   _config;
    std::uint64_t                             _deviceBytes{0};
};

/**
 * Open the checkpoint at `checkpointDir` (a directory holding the
 * safetensors shards, `model.safetensors.index.json`, and
 * `hf_quant_config.json`), upload every tensor to the device via
 * `uploader`, and assemble + validate every quantised weight.
 *
 * Throws std::runtime_error on a missing/malformed checkpoint or config, a
 * weight that fails `validateWeightLayout`, or an upload failure surfaced by
 * the uploader.
 */
[[nodiscard]] NvFp4Model loadNvfp4Model(const std::string& checkpointDir,
                                        DeviceUploader&    uploader);

} // namespace mimirmind::runtime::nvfp4