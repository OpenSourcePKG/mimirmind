// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/CudaModule.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"

#include "core/log/Log.hpp"

#include <cstddef>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

namespace mimirmind::core::cuda {

namespace {

inline void cudaDriverCheck(const char* call, CUresult code) {
    if (code != CUDA_SUCCESS) {
        throw CudaDriverError(call, code);
    }
}

std::vector<std::byte> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("CudaModule::fromFile: cannot open " + path);
    }
    const std::streamsize sz = f.tellg();
    if (sz <= 0) {
        throw std::runtime_error("CudaModule::fromFile: empty file " + path);
    }
    f.seekg(0, std::ios::beg);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!f) {
        throw std::runtime_error("CudaModule::fromFile: short read on " + path);
    }
    return buf;
}

} // namespace

CudaModule::CudaModule(CudaContext& ctx, std::span<const std::byte> blob)
    : _ctx(&ctx) {
    if (blob.empty()) {
        throw CudaDriverError("CudaModule(empty blob)", CUDA_ERROR_INVALID_VALUE);
    }
    // cuModuleLoadData disambiguates PTX vs cubin from the leading
    // bytes. PTX needs to be null-terminated in memory; nvcc --ptx
    // emits a trailing 0x0A newline but not a nul, so ensure our
    // buffer has a nul past the end when the blob looks like text.
    // For binary cubin the byte pattern is header-tagged and the nul
    // check is irrelevant.
    cudaDriverCheck("cuModuleLoadData",
                    cuModuleLoadData(&_module, blob.data()));
    MM_LOG_DEBUG("CudaModule", "loaded {} bytes onto device #{}",
                 blob.size(), _ctx->cudaDeviceIndex());
}

CudaModule CudaModule::fromFile(CudaContext& ctx, const std::string& path) {
    auto blob = readFile(path);
    // For text PTX files, append a nul byte for safety — cuModuleLoadData
    // expects C-string PTX.
    if (!blob.empty() && static_cast<char>(blob.back()) != '\0') {
        blob.push_back(std::byte{0});
    }
    return CudaModule{ctx, std::span<const std::byte>{blob.data(), blob.size()}};
}

CudaModule::~CudaModule() noexcept {
    destroy();
}

CudaModule::CudaModule(CudaModule&& other) noexcept
    : _ctx   (other._ctx),
      _module(other._module) {
    other._ctx    = nullptr;
    other._module = nullptr;
}

CudaModule& CudaModule::operator=(CudaModule&& other) noexcept {
    if (this != &other) {
        destroy();
        _ctx    = other._ctx;
        _module = other._module;
        other._ctx    = nullptr;
        other._module = nullptr;
    }
    return *this;
}

void CudaModule::destroy() noexcept {
    if (_module != nullptr) {
        const CUresult rc = cuModuleUnload(_module);
        if (rc != CUDA_SUCCESS) {
            const char* msg = nullptr;
            if (cuGetErrorString(rc, &msg) != CUDA_SUCCESS || msg == nullptr) {
                msg = "unknown";
            }
            MM_LOG_WARN("CudaModule", "cuModuleUnload failed: {}", msg);
        }
        _module = nullptr;
    }
    _ctx = nullptr;
}

CudaKernel CudaModule::getFunction(std::string_view name) {
    const std::string zname{name};
    CUfunction fn = nullptr;
    cudaDriverCheck("cuModuleGetFunction",
                    cuModuleGetFunction(&fn, _module, zname.c_str()));
    return CudaKernel{*this, fn, zname};
}

} // namespace mimirmind::core::cuda