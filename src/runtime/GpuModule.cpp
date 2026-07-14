// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/GpuModule.hpp"

#include "core/l0/L0Context.hpp"
#include "core/log/Log.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime {

using ::mimirmind::core::l0::L0Error;

namespace {

constexpr const char* kDefaultSpvDir = "/usr/local/share/mimirmind/spv";

std::filesystem::path resolveSpvPath(std::string_view spvDirOverride,
                                     std::string_view name) {
    const std::string filename = std::string{name} + ".spv";

    // 1. Config-supplied override (from runtime.spvDir in config.json).
    if (!spvDirOverride.empty()) {
        const std::filesystem::path p =
            std::filesystem::path{spvDirOverride} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    // 2. Production install location.
    {
        const std::filesystem::path p =
            std::filesystem::path{kDefaultSpvDir} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    // 3. Build-tree fallback (relative to cwd).
    for (auto rel : std::array<const char*, 3>{
             "build/spv", "../build/spv", "spv"}) {
        const std::filesystem::path p = std::filesystem::path{rel} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    throw std::runtime_error(
        "GpuModule: cannot find " + filename +
        " — set runtime.spvDir in config.json or install to " + kDefaultSpvDir);
}

} // namespace

GpuModule::GpuModule(L0Context& ctx, std::string_view name)
    : _ctx{ctx}, _name{name}
{
    const auto path = resolveSpvPath(_ctx.spvDirOverride(), _name);
    auto bytes      = readSpv(_ctx.spvDirOverride(), _name);

    MM_LOG_INFO("gpu", "loading module '{}' from {} ({} bytes)",
                _name, path.string(), bytes.size());

    ze_module_desc_t desc{};
    desc.stype        = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format       = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.inputSize    = bytes.size();
    desc.pInputModule = bytes.data();
    desc.pBuildFlags  = nullptr;
    desc.pConstants   = nullptr;

    ze_module_build_log_handle_t buildLog = nullptr;

    const ze_result_t r = zeModuleCreate(_ctx.context(),
                                         _ctx.device(),
                                         &desc,
                                         &_module,
                                         &buildLog);
    if (r != ZE_RESULT_SUCCESS) {
        std::string logText;
        if (buildLog != nullptr) {
            std::size_t logSize = 0;
            zeModuleBuildLogGetString(buildLog, &logSize, nullptr);
            logText.resize(logSize);
            zeModuleBuildLogGetString(buildLog, &logSize, logText.data());
            zeModuleBuildLogDestroy(buildLog);
        }
        MM_LOG_ERROR("gpu",
                     "zeModuleCreate failed for '{}': {} (0x{:x}) — log: {}",
                     _name, L0Context::resultToString(r),
                     static_cast<unsigned>(r), logText);
        throw L0Error("zeModuleCreate(" + _name + ")", r);
    }
    if (buildLog != nullptr) {
        zeModuleBuildLogDestroy(buildLog);
    }

    MM_LOG_INFO("gpu", "module '{}' built OK — handle={}",
                _name, static_cast<const void*>(_module));
}

GpuModule::~GpuModule() {
    for (auto& [n, k] : _kernels) {
        if (k != nullptr) {
            zeKernelDestroy(k);
        }
    }
    if (_module != nullptr) {
        zeModuleDestroy(_module);
        _module = nullptr;
    }
}

GpuKernel GpuModule::kernel(const char* kernelName) {
    if (const auto it = _kernels.find(kernelName); it != _kernels.end()) {
        return GpuKernel{it->second};
    }
    ze_kernel_desc_t desc{};
    desc.stype       = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    desc.pKernelName = kernelName;
    ze_kernel_handle_t k = nullptr;
    const ze_result_t r = zeKernelCreate(_module, &desc, &k);
    if (r != ZE_RESULT_SUCCESS) {
        MM_LOG_ERROR("gpu",
                     "zeKernelCreate failed for '{}' in '{}': {} (0x{:x})",
                     kernelName, _name,
                     L0Context::resultToString(r), static_cast<unsigned>(r));
        throw L0Error("zeKernelCreate(" + std::string{kernelName} + ")", r);
    }
    _kernels.emplace(kernelName, k);
    MM_LOG_DEBUG("gpu", "kernel '{}/{}' created — handle={}",
                 _name, kernelName, static_cast<const void*>(k));
    return GpuKernel{k};
}

std::vector<std::uint8_t> GpuModule::readSpv(std::string_view spvDirOverride,
                                             std::string_view name) {
    const auto path = resolveSpvPath(spvDirOverride, name);
    std::ifstream f{path, std::ios::binary | std::ios::ate};
    if (!f.is_open()) {
        throw std::runtime_error("GpuModule: cannot open " + path.string());
    }
    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(sz));
    if (!f.read(reinterpret_cast<char*>(bytes.data()), sz)) {
        throw std::runtime_error("GpuModule: read failed: " + path.string());
    }
    return bytes;
}

} // namespace mimirmind::runtime