// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace mimirmind::core::modelopt {

/**
 * HF (ModelOpt safetensors) -> GGUF tensor-name mapping for the `qwen35moe`
 * architecture, so the NVFP4 loader can materialise weights the existing
 * GGUF-convention arch backend addresses.
 *
 * Established authoritatively by comparing the running GGUF checkpoint
 * (unsloth Qwen3.6-35B-A3B UD-Q4_K_XL, which the qwen35moe backend already
 * runs end-to-end) tensor names + shapes against the HF safetensors names +
 * shapes. Key findings:
 *   - Every weight is BYTE-layout-compatible: HF PyTorch `[out, in]`
 *     row-major equals GGUF `[in(ne0), out(ne1)]`; even `conv1d [8192,1,4]`
 *     equals GGUF `[4, 8192]` byte-for-byte. So the only real data transform
 *     is stacking per-expert weights into the GGUF 3-D expert tensor.
 *   - unsloth reuses the `attn_qkv` / `attn_gate` GGUF names for the
 *     DeltaNet (`linear_attn`) `in_proj_qkv` / `in_proj_z`; the full-attn
 *     layers use `attn_q` (the FULL gated `q_proj [8192,2048]`, NOT split).
 *   - Full-attn vs DeltaNet layers are selected by `full_attention_interval`
 *     (=4: layers 3,7,...,39 are full-attn, the rest DeltaNet).
 */

/// Transform applied when materialising a GGUF tensor from its HF source(s).
enum class WeightXform : std::uint8_t {
    Direct,       ///< dequantise the one HF tensor; bytes map 1:1
    StackExperts, ///< dequantise each of the N experts into consecutive ne2 slices
};

/// One GGUF tensor and the HF tensor it comes from. `hfSuffix` is relative to
/// the per-layer prefix `model.language_model.layers.<L>.`, except in the
/// top-level table where it is the full HF name. For `StackExperts`,
/// `hfSuffix` contains the literal `{E}` placeholder for the expert index.
struct GgufTensorSource {
    std::string_view ggufSuffix; ///< e.g. "attn_q.weight" (block) or "output.weight" (top-level)
    std::string_view hfSuffix;   ///< e.g. "self_attn.q_proj.weight", or full name at top level
    WeightXform      xform;
};

/// Per-block tensors of a full-attention layer.
[[nodiscard]] std::span<const GgufTensorSource> qwen35moeFullAttnTensors() noexcept;

/// Per-block tensors of a DeltaNet (linear_attn) layer.
[[nodiscard]] std::span<const GgufTensorSource> qwen35moeDeltaNetTensors() noexcept;

/// Per-block MoE tensors (router, routed experts, shared expert) — present on
/// every layer regardless of attention type.
[[nodiscard]] std::span<const GgufTensorSource> qwen35moeMoeTensors() noexcept;

/// Per-block DENSE MLP tensors (SwiGLU) for the dense qwen3_5 variant
/// (Qwen3.8-27B) — the `mlp.{gate,up,down}_proj` triple that replaces the MoE
/// router + expert banks. Same GGUF names (ffn_gate/up/down) the dense backend
/// (Qwen3_5DenseBackend::runFfn) addresses.
[[nodiscard]] std::span<const GgufTensorSource> qwen35DenseMlpTensors() noexcept;

/// Model-level tensors (embeddings, final norm, lm_head). `hfSuffix` and
/// `ggufSuffix` are the full names here.
[[nodiscard]] std::span<const GgufTensorSource> qwen35moeTopLevelTensors() noexcept;

/// True if layer `layer` (0-based) is a full-attention layer under the given
/// interval (default 4). `(layer + 1) % interval == 0`, matching the GGUF
/// (layers 3,7,...).
[[nodiscard]] bool qwen35moeIsFullAttnLayer(int layer, int interval = 4) noexcept;

/// Build the full HF tensor name for a per-block source: prefix
/// `model.language_model.layers.<layer>.` + `hfSuffix`, with `{E}` replaced by
/// `expert` when `expert >= 0`.
[[nodiscard]] std::string qwen35moeHfName(std::string_view hfSuffix, int layer,
                                          int expert = -1);

/// The four MTP (nextn) head projections. `ggufSuffix` is relative to the MTP
/// block prefix `blk.<numLayers>.` (e.g. "nextn.eh_proj.weight"); `hfSuffix`
/// is relative to the top-level MTP prefix `mtp.` (e.g. "fc.weight"). The MTP
/// transformer block itself reuses the full-attention + MoE tables.
[[nodiscard]] std::span<const GgufTensorSource> qwen35moeNextnTensors() noexcept;

/// Build the full HF name for a tensor inside the MTP transformer block:
/// prefix `mtp.layers.0.` + `hfSuffix`, with `{E}` replaced by `expert` when
/// `expert >= 0`. (The HF checkpoint numbers the single MTP block as
/// `mtp.layers.0`, independent of the main stack depth.)
[[nodiscard]] std::string qwen35moeMtpHfName(std::string_view hfSuffix,
                                             int expert = -1);

} // namespace mimirmind::core::modelopt