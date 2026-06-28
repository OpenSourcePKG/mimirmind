#pragma once

#include <level_zero/ze_api.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mimirmind::runtime {

class L0Context;

/**
 * A single SPIR-V module loaded into a Level Zero context. Holds the
 * ze_module_handle_t plus a cache of `ze_kernel_handle_t` looked up by
 * kernel name.
 *
 * Loaded from a `.spv` file on disk (produced by ocloc at build time).
 * The path is resolved against `MIMIRMIND_SPV_DIR` if set, else
 * `/usr/local/share/mimirmind/spv`, else the build-tree fallback.
 */
class GpuModule {
public:
    /// Load `<spvDirOrEnv>/<name>.spv` into the context's L0 module.
    GpuModule(L0Context& ctx, std::string_view name);
    ~GpuModule();

    GpuModule(const GpuModule&)            = delete;
    GpuModule& operator=(const GpuModule&) = delete;
    GpuModule(GpuModule&&)                 = delete;
    GpuModule& operator=(GpuModule&&)      = delete;

    /// Look up (or first-time-create) a kernel by its `__kernel` symbol
    /// name. Throws L0Error if the symbol isn't in this module.
    [[nodiscard]] ze_kernel_handle_t kernel(const char* kernelName);

    [[nodiscard]] std::string_view name() const noexcept { return _name; }

private:
    static std::vector<std::uint8_t> readSpv(std::string_view name);

    L0Context&                                       _ctx;
    std::string                                      _name;
    ze_module_handle_t                               _module{nullptr};
    std::unordered_map<std::string, ze_kernel_handle_t> _kernels;
};

} // namespace mimirmind::runtime