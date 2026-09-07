// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/NvFp4Model.hpp"

#include "core/log/Log.hpp"
#include "core/modelopt/ModelOptQuant.hpp"
#include "core/modelopt/ModelOptWeightAssembler.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <span>
#include <vector>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mimirmind::runtime::nvfp4 {

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("nvfp4 load: " + msg);
}

std::string readTextFile(const fs::path& p) {
    std::ifstream f(p);
    if (!f) {
        fail("cannot read " + p.string());
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool endsWith(std::string_view s, std::string_view suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

} // namespace

const NvFp4DeviceTensor* NvFp4Model::find(std::string_view name) const noexcept {
    const auto it = _tensors.find(std::string(name));
    return it == _tensors.end() ? nullptr : &it->second;
}

const NvFp4DeviceWeight* NvFp4Model::weight(std::string_view module) const noexcept {
    const auto it = _weights.find(std::string(module));
    return it == _weights.end() ? nullptr : &it->second;
}

void NvFp4Model::releaseTensor(const std::string& name) noexcept {
    const auto it = _bufIdx.find(name);
    if (it == _bufIdx.end()) {
        return;
    }
    static const bool kDiag = std::getenv("MIMIRMIND_Q4E_DIAG") != nullptr;
    const std::size_t idx = it->second;
    if (kDiag) {
        MM_LOG_INFO("q4ediag", "release '{}' ptr={} bytes={} refs={}",
                    name, _buffers[idx].get(), _buffers[idx].bytes(),
                    idx < _bufRefs.size() ? _bufRefs[idx] : 1);
    }
    // Slab members share one buffer; the device memory goes away with the
    // LAST member released (refcount), not the first.
    if (idx >= _bufRefs.size() || _bufRefs[idx] <= 1
        || --_bufRefs[idx] == 0) {
        _buffers[idx] = compute::ComputeBuffer{};   // free device memory
    }
    _bufIdx.erase(it);
    // The _tensors entry stays (its devPtr is now dangling); the caller
    // guarantees it will not be read again (5.27 streaming repack).
}

namespace {

/// Copy rows of `src` (rowBytes each) into a new buffer following
/// `dstToSrcRow` (dst row i takes src row dstToSrcRow[i]). Empty on a
/// size mismatch so callers can fail the whole split loudly.
std::vector<std::uint8_t>
permuteRows(std::span<const std::uint8_t>   src,
            std::size_t                     rowBytes,
            const std::vector<std::size_t>& dstToSrcRow) {
    std::vector<std::uint8_t> out;
    if (rowBytes == 0) {
        return out;
    }
    const std::size_t srcRows = src.size() / rowBytes;
    if (src.size() % rowBytes != 0) {
        return out;
    }
    out.resize(dstToSrcRow.size() * rowBytes);
    for (std::size_t i = 0; i < dstToSrcRow.size(); ++i) {
        const std::size_t s = dstToSrcRow[i];
        if (s >= srcRows) {
            return {};
        }
        std::memcpy(out.data() + i * rowBytes,
                    src.data() + s * rowBytes, rowBytes);
    }
    return out;
}

} // namespace

bool normalizeCompressedTensorsCheckpoint(safetensors::SafetensorsModel& sm) {
    // Dialect probe: a packed weight under the un-prefixed layer namespace
    // only exists in llm-compressor "nvfp4-pack-quantized" exports.
    bool ct = false;
    for (const auto* t : sm.tensors()) {
        if (t->name.rfind("model.layers.", 0) == 0
            && t->name.size() > 14
            && t->name.compare(t->name.size() - 14, 14, ".weight_packed") == 0) {
            ct = true;
            break;
        }
    }
    if (!ct) {
        return false;
    }

    sm.normalizeNames([](const std::string& n) {
        std::string s = n;
        if (s.rfind("model.layers.", 0) == 0) {
            s = "model.language_model.layers." + s.substr(13);
        } else if (s.rfind("model.embed_tokens", 0) == 0
                   || s.rfind("model.norm", 0) == 0) {
            s = "model.language_model." + s.substr(6);
        }
        const auto swapSuffix = [&s](const char* from, const char* to) {
            const std::size_t fl = std::char_traits<char>::length(from);
            if (s.size() >= fl && s.compare(s.size() - fl, fl, from) == 0) {
                s = s.substr(0, s.size() - fl) + to;
            }
        };
        swapSuffix(".weight_packed", ".weight");
        swapSuffix(".weight_global_scale", ".weight_scale_2");
        return s;
    });

    // compressed-tensors stores the RECIPROCAL of ModelOpt's direct global
    // scale — serve inverted values via byte overrides (mmap is read-only).
    std::size_t inverted = 0;
    for (const auto* t : sm.tensors()) {
        const std::string& n = t->name;
        constexpr const char* kSuf = ".weight_scale_2";
        constexpr std::size_t kSufLen = 15;
        if (n.size() < kSufLen
            || n.compare(n.size() - kSufLen, kSufLen, kSuf) != 0) {
            continue;
        }
        if (t->dtype != safetensors::SafetensorsDtype::F32 || t->nbytes != 4) {
            continue;
        }
        const auto b = sm.tensorBytes(n);
        if (b.size() != 4) {
            continue;
        }
        float v = 0.0F;
        std::memcpy(&v, b.data(), 4);
        const float inv = (v != 0.0F) ? 1.0F / v : 0.0F;
        std::vector<std::uint8_t> nb(4);
        std::memcpy(nb.data(), &inv, 4);
        sm.overrideTensorBytes(n, std::move(nb));
        ++inverted;
    }
    // --- GDN fused-projection splits ------------------------------------
    // Qwen3-Next fuses the GDN input projections, and the fused row order is
    // PER-KEY-HEAD INTERLEAVED, not flat: for each key head g,
    //   in_proj_qkvz rows = [q_g(headK), k_g(headK), v_g(r*headV), z_g(r*headV)]
    //   in_proj_ba   rows = [b_g(r), a_g(r)]        with r = nV/nK
    // (HF modeling_qwen3_next fix_query_key_value_ordering). The qwen3_5
    // materializer expects flat separate tensors ([q;k;v], [z], [b], [a]), so
    // each split is a row PERMUTATION, served via byte overrides on top of
    // deriveRowSlice's shape metadata. z's total row count (= valueDim)
    // derives from out_proj's input width (packed cols * 2); nV from ba's
    // rows (= 2*nV); headK is assumed == headV (true for the whole
    // qwen3-next family) and cross-checked against the fused row total.
    std::vector<std::string> qkvzW, baW;
    for (const auto* t : sm.tensors()) {
        const std::string& n = t->name;
        const auto ends = [&n](const char* suf) {
            const std::size_t l = std::char_traits<char>::length(suf);
            return n.size() >= l && n.compare(n.size() - l, l, suf) == 0;
        };
        if (ends(".linear_attn.in_proj_qkvz.weight")) qkvzW.push_back(n);
        else if (ends(".linear_attn.in_proj_ba.weight")) baW.push_back(n);
    }
    std::size_t splits = 0;
    std::vector<std::string> toRemove;
    // Per-module (nK, r=nV/nK) from the qkvz split, consumed by the ba split.
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> gdnGeo;
    for (const auto& w : qkvzW) {
        const std::string base = w.substr(0, w.size() - 7);   // …in_proj_qkvz
        const std::string mod =
            base.substr(0, base.size() - std::strlen("in_proj_qkvz"));
        const auto* wt = sm.find(w);
        const auto* op = sm.find(mod + "out_proj.weight");
        if (wt == nullptr || op == nullptr
            || wt->shape.size() != 2 || op->shape.size() != 2) {
            continue;
        }
        const bool opPacked = op->dtype == safetensors::SafetensorsDtype::U8;
        const std::uint64_t zRows = op->shape[1] * (opPacked ? 2 : 1);
        const std::uint64_t rows  = wt->shape[0];
        if (zRows == 0 || zRows >= rows) {
            continue;
        }
        const std::uint64_t qkvRows   = rows - zRows;
        const std::uint64_t wBytes    = wt->nbytes;

        // Head geometry from shapes (see the layout note above): valueDim =
        // zRows; nV from ba's rows; headK assumed == headV, cross-checked.
        const auto* ba = sm.find(mod + "in_proj_ba.weight");
        if (ba == nullptr || ba->shape.empty() || ba->shape[0] % 2 != 0) {
            continue;
        }
        const std::uint64_t valueDim = zRows;
        const std::uint64_t nV       = ba->shape[0] / 2;
        if (nV == 0 || valueDim % nV != 0 || (qkvRows - valueDim) % 2 != 0) {
            continue;
        }
        const std::uint64_t headV  = valueDim / nV;
        const std::uint64_t headK  = headV;              // qwen3-next family
        const std::uint64_t keyDim = (qkvRows - valueDim) / 2;
        if (headK == 0 || keyDim % headK != 0) {
            continue;
        }
        const std::uint64_t nK = keyDim / headK;
        if (nK == 0 || nV % nK != 0) {
            continue;
        }
        const std::uint64_t r      = nV / nK;
        const std::uint64_t rV     = r * headV;          // v/z rows per group
        const std::uint64_t grpRows = 2 * headK + 2 * rV;
        if (nK * grpRows != rows || rows == 0 || wBytes % rows != 0) {
            MM_LOG_WARN("nvfp4",
                        "GDN split: fused row total {} != nK {} * group {} "
                        "for '{}' — skipping (headK==headV assumption?)",
                        rows, nK, grpRows, w);
            continue;
        }

        // dst->src row maps for the flat [q;k;v] / [z] targets.
        std::vector<std::size_t> qkvMap(qkvRows);
        std::vector<std::size_t> zMap(zRows);
        for (std::uint64_t g = 0; g < nK; ++g) {
            const std::uint64_t s = g * grpRows;
            for (std::uint64_t j = 0; j < headK; ++j) {
                qkvMap[g * headK + j]          = s + j;               // q
                qkvMap[keyDim + g * headK + j] = s + headK + j;       // k
            }
            for (std::uint64_t j = 0; j < rV; ++j) {
                qkvMap[2 * keyDim + g * rV + j] = s + 2 * headK + j;  // v
                zMap[g * rV + j]                = s + 2 * headK + rV + j;
            }
        }

        bool ok = true;
        ok = ok && sm.deriveRowSlice(w, mod + "in_proj_qkv.weight", 0, qkvRows);
        ok = ok && sm.deriveRowSlice(w, mod + "in_proj_z.weight", qkvRows, zRows);
        if (ok) {
            const auto srcW = sm.tensorBytes(w);
            const std::size_t wRowBytes =
                static_cast<std::size_t>(wBytes / rows);
            auto qkvB = permuteRows(srcW, wRowBytes, qkvMap);
            auto zB   = permuteRows(srcW, wRowBytes, zMap);
            ok = !qkvB.empty() && !zB.empty();
            if (ok) {
                sm.overrideTensorBytes(mod + "in_proj_qkv.weight",
                                       std::move(qkvB));
                sm.overrideTensorBytes(mod + "in_proj_z.weight",
                                       std::move(zB));
            }
        }
        if (sm.find(base + ".weight_scale") != nullptr) {
            ok = ok && sm.deriveRowSlice(base + ".weight_scale",
                                         mod + "in_proj_qkv.weight_scale",
                                         0, qkvRows);
            ok = ok && sm.deriveRowSlice(base + ".weight_scale",
                                         mod + "in_proj_z.weight_scale",
                                         qkvRows, zRows);
            if (ok) {
                const auto srcS = sm.tensorBytes(base + ".weight_scale");
                if (srcS.size() % rows == 0 && !srcS.empty()) {
                    const std::size_t sRowBytes = srcS.size() / rows;
                    auto qkvS = permuteRows(srcS, sRowBytes, qkvMap);
                    auto zS   = permuteRows(srcS, sRowBytes, zMap);
                    ok = !qkvS.empty() && !zS.empty();
                    if (ok) {
                        sm.overrideTensorBytes(mod + "in_proj_qkv.weight_scale",
                                               std::move(qkvS));
                        sm.overrideTensorBytes(mod + "in_proj_z.weight_scale",
                                               std::move(zS));
                    }
                } else {
                    ok = false;
                }
            }
            ok = ok && sm.duplicateTensorAs(base + ".weight_scale_2",
                                            mod + "in_proj_qkv.weight_scale_2");
            ok = ok && sm.duplicateTensorAs(base + ".weight_scale_2",
                                            mod + "in_proj_z.weight_scale_2");
            const auto ovb = sm.tensorBytes(base + ".weight_scale_2");
            if (ovb.size() == 4) {
                sm.overrideTensorBytes(mod + "in_proj_qkv.weight_scale_2",
                                       {ovb.begin(), ovb.end()});
                sm.overrideTensorBytes(mod + "in_proj_z.weight_scale_2",
                                       {ovb.begin(), ovb.end()});
            }
            toRemove.push_back(base + ".weight_scale");
            toRemove.push_back(base + ".weight_scale_2");
        }
        if (ok) {
            gdnGeo.emplace(mod, std::make_pair(nK, r));
            toRemove.push_back(w);
            ++splits;
        }
    }
    for (const auto& w : baW) {
        const std::string mod =
            w.substr(0, w.size() - std::strlen("in_proj_ba.weight"));
        const auto* wt = sm.find(w);
        if (wt == nullptr || wt->shape.empty() || wt->shape[0] % 2 != 0) {
            continue;
        }
        const std::uint64_t half   = wt->shape[0] / 2;   // = nV
        const std::uint64_t rows   = wt->shape[0];
        const std::uint64_t wBytes = wt->nbytes;
        const auto geoIt = gdnGeo.find(mod);
        if (geoIt == gdnGeo.end() || wBytes % rows != 0) {
            continue;   // qkvz sibling failed its split — leave fused, fail loud
        }
        const std::uint64_t nK = geoIt->second.first;
        const std::uint64_t r  = geoIt->second.second;
        if (nK * r != half) {
            continue;
        }
        // Fused rows are per-key-head [b_g(r), a_g(r)]; targets are flat.
        std::vector<std::size_t> bMap(half);
        std::vector<std::size_t> aMap(half);
        for (std::uint64_t g = 0; g < nK; ++g) {
            for (std::uint64_t j = 0; j < r; ++j) {
                bMap[g * r + j] = g * 2 * r + j;
                aMap[g * r + j] = g * 2 * r + r + j;
            }
        }
        bool ok = true;
        ok = ok && sm.deriveRowSlice(w, mod + "in_proj_b.weight", 0, half);
        ok = ok && sm.deriveRowSlice(w, mod + "in_proj_a.weight", half, half);
        if (ok) {
            const auto srcW = sm.tensorBytes(w);
            const std::size_t rowBytes =
                static_cast<std::size_t>(wBytes / rows);
            auto bB = permuteRows(srcW, rowBytes, bMap);
            auto aB = permuteRows(srcW, rowBytes, aMap);
            ok = !bB.empty() && !aB.empty();
            if (ok) {
                sm.overrideTensorBytes(mod + "in_proj_b.weight", std::move(bB));
                sm.overrideTensorBytes(mod + "in_proj_a.weight", std::move(aB));
            }
        }
        if (ok) {
            toRemove.push_back(w);
            ++splits;
        }
    }
    for (const auto& n : toRemove) {
        sm.removeTensor(n);
    }
    sm.rebuildIndexes();

    MM_LOG_INFO("nvfp4",
                "compressed-tensors checkpoint normalised to ModelOpt dialect "
                "({} global scales inverted, {} fused GDN projections split)",
                inverted, splits);
    return true;
}

std::string syntheticCtHfQuantConfigJson() {
    // Uniform NVFP4, group 16, no excludes: the checkpoint-truth guards in
    // the assembler/materializer route BF16 (unquantised) tensors correctly
    // without an explicit exclude list.
    return R"({"producer":{"name":"modelopt-ct-shim","version":"0"},)"
           R"("quantization":{"quant_algo":"NVFP4","group_size":16,)"
           R"("exclude_modules":[]}})";
}

NvFp4Model loadNvfp4Model(const std::string& checkpointDir, DeviceUploader& uploader) {
    safetensors::SafetensorsModel sm;
    sm.open(checkpointDir); // throws on a missing/malformed checkpoint
    return loadNvfp4Model(sm, checkpointDir, uploader);
}

NvFp4Model loadNvfp4Model(safetensors::SafetensorsModel& sm,
                          const std::string&             configDir,
                          DeviceUploader&                uploader) {
    const fs::path dir{configDir};

    // ModelOpt checkpoints (qwen35moe) ship an hf_quant_config.json driving the
    // assembler's per-module scheme lookup. compressed-tensors checkpoints
    // (Gemma-4 "nvfp4-pack-quantized") keep the quant config inside config.json
    // and expose no hf_quant_config.json — their materializer resolves every
    // scale sidecar by explicit name from the raw _tensors map, so the assembled
    // _weights map is not needed. Tolerate the missing file: upload all tensors,
    // skip the assembly pass.
    const fs::path quantCfg = dir / "hf_quant_config.json";
    bool haveQuantCfg = fs::is_regular_file(quantCfg);

    // llm-compressor "nvfp4-pack-quantized" exports (Qwen3-Coder-Next) ship
    // no hf_quant_config.json and a different tensor-name dialect; normalise
    // them to the ModelOpt convention and synthesise the equivalent config.
    const bool ctNormalized = normalizeCompressedTensorsCheckpoint(sm);

    NvFp4Model out;
    if (haveQuantCfg) {
        out._config = modelopt::HfQuantConfig::parse(readTextFile(quantCfg));
    } else if (ctNormalized) {
        out._config = modelopt::HfQuantConfig::parse(syntheticCtHfQuantConfigJson());
        haveQuantCfg = true;
    }

    // Per-expert checkpoints (Qwen3.5-122B-A10B, qwen4_exp) ship every routed
    // expert as its own tensor set — ~74k device allocations whose interleaved
    // 1.5 MiB frees during the streaming repack corrupt live neighbouring
    // managed mappings on GB10 (driver-level; our own free bookkeeping showed
    // no overlap, the mapping still died at layer 7). Consolidate those
    // sources into one slab per (layer, projection, tensor kind): 256 members
    // share one allocation, released as a whole once every member was
    // consumed. ~144 large frees instead of ~74k interleaved small ones.
    auto expertSlabKey = [](const std::string& n) -> std::string {
        static const char* kPre = "model.language_model.layers.";
        if (n.rfind(kPre, 0) != 0) return {};
        const auto ex = n.find(".mlp.experts.");
        if (ex == std::string::npos) return {};
        const auto eBeg = ex + 13;
        const auto dot  = n.find('.', eBeg);
        if (dot == std::string::npos || dot == eBeg) return {};
        for (auto i = eBeg; i < dot; ++i) {
            if (n[i] < '0' || n[i] > '9') return {};
        }
        const std::string tail = n.substr(dot + 1);
        // One slab per (layer, projection) holding weight + weight_scale +
        // weight_scale_2 of all experts TOGETHER (~453 MiB in one 512 MiB
        // pool block instead of a 402 MiB weight slab + managed scales):
        // less quantisation padding, and the scale frees become
        // pool-recyclable too. input_scale stays OUT — it is never part of
        // releaseStepSources, and a never-released member would pin the
        // slab's refcount above zero forever.
        for (const char* k : {"gate_proj.weight", "up_proj.weight",
                              "down_proj.weight", "gate_proj.weight_scale",
                              "up_proj.weight_scale", "down_proj.weight_scale",
                              "gate_proj.weight_scale_2", "up_proj.weight_scale_2",
                              "down_proj.weight_scale_2"}) {
            if (tail == k) {
                const auto projDot = tail.find('.');
                return n.substr(0, ex) + "|" + tail.substr(0, projDot);
            }
        }
        return {};
    };
    constexpr std::size_t kSlabAlign = 256;
    auto slabPad = [](std::size_t b) {
        return (b + kSlabAlign - 1) & ~(kSlabAlign - 1);
    };
    struct Slab {
        std::size_t   bytes   = 0;          // planned total
        std::size_t   cursor  = 0;          // upload offset
        std::uint32_t members = 0;
        std::size_t   bufIdx  = SIZE_MAX;   // _buffers slot once allocated
        void*         base    = nullptr;
    };
    std::map<std::string, Slab> slabs;

    // The single-nextn MTP walk is skipped for per-expert MTP layouts
    // (Nvfp4Loader forces mtpLayers=0), so those checkpoints' mtp.* tensors
    // are never read — do not spend device memory uploading them (~4.6 GiB
    // + thousands of allocations on the 122B). Stacked-MTP checkpoints
    // (qwen3.6-35B) keep uploading mtp.* unchanged.
    const bool perExpertMtp =
        sm.find("mtp.layers.0.mlp.experts.0.gate_proj.weight") != nullptr;

    // --- pre-scan: size the expert-source slabs --------------------------
    for (const safetensors::SafetensorsTensor* t : sm.tensors()) {
        if (t->nbytes == 0) continue;
        const std::string key = expertSlabKey(t->name);
        if (key.empty()) continue;
        auto& s = slabs[key];
        s.bytes += slabPad(t->nbytes);
        ++s.members;
    }

    // --- upload every tensor to the device -------------------------------
    for (const safetensors::SafetensorsTensor* t : sm.tensors()) {
        // 5.27 I-2 (qwen4_exp): do NOT upload tensor classes we don't load
        // resident — the PLE n-gram table (~48 GiB, served off-VRAM via mmap in
        // I-4) and the VL vision tower (text-only bring-up). The materialization
        // plan already ignores them, but this upload loop is unconditional, so
        // without the skip the 48 GiB PLE would transiently blow up VRAM before
        // the plan discards it. These names exist only on qwen4_exp checkpoints,
        // so the skip is a no-op for every other model.
        if (t->name.find(".ple.ple_embedding") != std::string::npos ||
            t->name.rfind("model.visual.", 0) == 0) {
            continue;
        }
        if (perExpertMtp && t->name.rfind("mtp.", 0) == 0) {
            continue;
        }
        NvFp4DeviceTensor dev;
        dev.dtype = t->dtype;
        dev.shape = t->shape;
        dev.nbytes = t->nbytes;
        if (t->nbytes != 0) {
            const auto bytes = sm.tensorBytes(t->name);
            if (bytes.size() != t->nbytes) {
                fail("tensor '" + t->name + "' byte span size mismatch");
            }
            const std::string key = expertSlabKey(t->name);
            if (!key.empty()) {
                auto& s = slabs[key];
                if (s.bufIdx == SIZE_MAX) {
                    compute::ComputeBuffer slab = uploader.allocate(s.bytes);
                    if (slab.get() == nullptr) {
                        fail("slab alloc failed for '" + key + "' ("
                             + std::to_string(s.bytes >> 20) + " MiB)");
                    }
                    s.base   = slab.get();
                    s.bufIdx = out._buffers.size();
                    out._buffers.push_back(std::move(slab));
                    out._bufRefs.push_back(s.members);
                }
                auto* dst = static_cast<std::uint8_t*>(s.base) + s.cursor;
                uploader.uploadHostBytes(dst, bytes.data(), t->nbytes);
                s.cursor += slabPad(t->nbytes);
                dev.devPtr = dst;
                out._bufIdx.emplace(t->name, s.bufIdx);
                out._deviceBytes += t->nbytes;
            } else {
                compute::ComputeBuffer buf = uploader.allocate(t->nbytes);
                uploader.uploadHostBytes(buf.get(), bytes.data(), t->nbytes);
                dev.devPtr = buf.get();
                out._bufIdx.emplace(t->name, out._buffers.size());
                out._buffers.push_back(std::move(buf));
                out._bufRefs.push_back(1);
                out._deviceBytes += t->nbytes;
            }
        }
        out._tensors.emplace(t->name, std::move(dev));
    }

    // --- assemble + validate every quantised weight (ModelOpt only) ------
    if (!haveQuantCfg) {
        return out; // compressed-tensors: tensors uploaded, no assembly needed
    }
    const modelopt::ModelOptWeightAssembler assembler(sm, out._config);
    for (const safetensors::SafetensorsTensor* t : sm.tensors()) {
        if (!endsWith(t->name, ".weight")) {
            continue;
        }
        if (perExpertMtp && t->name.rfind("mtp.", 0) == 0) {
            continue;   // not uploaded (see the upload-loop skip above)
        }
        const std::string base = t->name.substr(0, t->name.size() - 7); // strip ".weight"
        if (!assembler.isQuantized(base)) {
            continue;
        }

        // Host-side assemble validates dtypes/shapes against the scheme
        // descriptor (throws on any inconsistency); we keep the layout and
        // resolve each sub-tensor to its already-uploaded device tensor.
        const modelopt::ModelOptWeight hw = assembler.assemble(base);
        const modelopt::ModelOptSchemeInfo& info = modelopt::schemeInfo(hw.layout.scheme);

        const auto devOf = [&](const std::string& name) -> NvFp4DeviceTensor {
            const NvFp4DeviceTensor* d = out.find(name);
            if (d == nullptr) {
                fail("assembled weight '" + base + "' references un-uploaded tensor '" + name + "'");
            }
            return *d;
        };

        NvFp4DeviceWeight dw;
        dw.layout       = hw.layout;
        dw.packedWeight = devOf(base + ".weight");
        if (info.hasBlockScale)        dw.blockScale  = devOf(base + ".weight_scale");
        if (info.hasGlobalScale)       dw.globalScale = devOf(base + ".weight_scale_2");
        if (info.hasTensorWeightScale) dw.weightScale = devOf(base + ".weight_scale");
        if (info.hasInputScale)        dw.inputScale  = devOf(base + ".input_scale");

        out._weights.emplace(base, std::move(dw));
    }

    return out;
}

} // namespace mimirmind::runtime::nvfp4