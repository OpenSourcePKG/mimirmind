// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipKernel.hpp"

#include "core/log/Log.hpp"

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

namespace mimirmind::core::hip {

namespace {

inline void hipCheck(const char* call, hipError_t code) {
    if (code != hipSuccess) {
        throw HipError(call, code);
    }
}

std::vector<std::byte> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("HipModule::fromFile: cannot open " + path);
    }
    const std::streamsize sz = f.tellg();
    if (sz <= 0) {
        throw std::runtime_error("HipModule::fromFile: empty file " + path);
    }
    f.seekg(0, std::ios::beg);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!f) {
        throw std::runtime_error("HipModule::fromFile: short read on " + path);
    }
    return buf;
}

} // namespace

HipModule::HipModule(HipContext& ctx, std::span<const std::byte> hsacoData)
    : _ctx(&ctx) {
    if (hsacoData.empty()) {
        throw HipError("HipModule(empty hsaco)", hipErrorInvalidValue);
    }
    // hipModuleLoadData takes a raw pointer; the code-object header
    // carries its own size, no explicit length needed.
    hipCheck("hipModuleLoadData",
             hipModuleLoadData(&_module, hsacoData.data()));
    MM_LOG_DEBUG("HipModule", "loaded {} bytes onto device #{}",
                 hsacoData.size(), _ctx->hipDeviceIndex());
}

HipModule HipModule::fromFile(HipContext& ctx, const std::string& path) {
    const auto blob = readFile(path);
    return HipModule{ctx, std::span<const std::byte>{blob.data(), blob.size()}};
}

HipModule::~HipModule() noexcept {
    destroy();
}

HipModule::HipModule(HipModule&& other) noexcept
    : _ctx   (other._ctx),
      _module(other._module) {
    other._ctx    = nullptr;
    other._module = nullptr;
}

HipModule& HipModule::operator=(HipModule&& other) noexcept {
    if (this != &other) {
        destroy();
        _ctx    = other._ctx;
        _module = other._module;
        other._ctx    = nullptr;
        other._module = nullptr;
    }
    return *this;
}

void HipModule::destroy() noexcept {
    if (_module != nullptr) {
        const hipError_t rc = hipModuleUnload(_module);
        if (rc != hipSuccess) {
            MM_LOG_WARN("HipModule", "hipModuleUnload failed: {}",
                        hipGetErrorString(rc));
        }
        _module = nullptr;
    }
    _ctx = nullptr;
}

HipKernel HipModule::getKernel(std::string_view name) {
    // hipModuleGetFunction takes a null-terminated C string. std::span-
    // based string_view might not be null-terminated, so materialise.
    const std::string zname{name};
    hipFunction_t fn = nullptr;
    hipCheck("hipModuleGetFunction",
             hipModuleGetFunction(&fn, _module, zname.c_str()));
    return HipKernel{*this, fn, zname};
}

} // namespace mimirmind::core::hip