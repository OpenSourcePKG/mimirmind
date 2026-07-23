// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/ModelFormatResolver.hpp"

#include <filesystem>
#include <string>

namespace mimirmind::runtime::nvfp4 {

namespace fs = std::filesystem;

core::config::ModelFormat
resolveModelFormat(core::config::ModelFormat declared, std::string_view path) {
    using core::config::ModelFormat;
    if (declared != ModelFormat::Auto) {
        return declared;
    }

    std::error_code ec;
    const fs::path p{std::string(path)};

    if (fs::is_directory(p, ec)) {
        if (fs::is_regular_file(p / "model.safetensors.index.json", ec)
            || fs::is_regular_file(p / "model.safetensors", ec)) {
            return ModelFormat::Nvfp4;
        }
        return ModelFormat::Gguf;
    }

    if (p.extension() == ".safetensors") {
        return ModelFormat::Nvfp4;
    }
    return ModelFormat::Gguf;
}

} // namespace mimirmind::runtime::nvfp4