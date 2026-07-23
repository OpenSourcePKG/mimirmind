// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/NvFp4Model.hpp"

#include "core/modelopt/ModelOptQuant.hpp"
#include "core/modelopt/ModelOptWeightAssembler.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

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

NvFp4Model loadNvfp4Model(const std::string& checkpointDir, DeviceUploader& uploader) {
    const fs::path dir{checkpointDir};

    safetensors::SafetensorsModel sm;
    sm.open(checkpointDir); // throws on a missing/malformed checkpoint

    const fs::path quantCfg = dir / "hf_quant_config.json";
    if (!fs::is_regular_file(quantCfg)) {
        fail("missing hf_quant_config.json in " + checkpointDir);
    }

    NvFp4Model out;
    out._config = modelopt::HfQuantConfig::parse(readTextFile(quantCfg));

    // --- upload every tensor to the device -------------------------------
    for (const safetensors::SafetensorsTensor* t : sm.tensors()) {
        NvFp4DeviceTensor dev;
        dev.dtype = t->dtype;
        dev.shape = t->shape;
        dev.nbytes = t->nbytes;
        if (t->nbytes != 0) {
            const auto bytes = sm.tensorBytes(t->name);
            if (bytes.size() != t->nbytes) {
                fail("tensor '" + t->name + "' byte span size mismatch");
            }
            compute::ComputeBuffer buf = uploader.allocate(t->nbytes);
            uploader.uploadHostBytes(buf.get(), bytes.data(), t->nbytes);
            dev.devPtr = buf.get();
            out._buffers.push_back(std::move(buf));
            out._deviceBytes += t->nbytes;
        }
        out._tensors.emplace(t->name, std::move(dev));
    }

    // --- assemble + validate every quantised weight ----------------------
    const modelopt::ModelOptWeightAssembler assembler(sm, out._config);
    for (const safetensors::SafetensorsTensor* t : sm.tensors()) {
        if (!endsWith(t->name, ".weight")) {
            continue;
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