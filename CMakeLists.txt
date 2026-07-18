cmake_minimum_required(VERSION 3.21)
project(mimirmind
    VERSION 0.0.1
    DESCRIPTION "Standalone C++ inference engine for GGUF, multi-backend"
    LANGUAGES CXX
)

# ---- Backend selection ------------------------------------------------------
#
# Each GPU backend is opt-in at configure time. Default is Level Zero ON
# because Meteor Lake iGPU is today's production target; HIP defaults OFF
# because the implementation lives on an isolated branch (see the
# HW-Abstraktions-Strategie note, revision 2026-07-14). Configure with
#   -DMIMIRMIND_ENABLE_L0=OFF   to skip the entire Level Zero build
#                               (mimirmind binary, munin, spv_kernels,
#                                gpu_tests, prefill_bench, l0_ipc_testrig
#                                are omitted). Useful on hosts without
#                                Intel HW / ocloc / IGC libs.
#   -DMIMIRMIND_ENABLE_HIP=ON   placeholder — no HIP backend impl yet,
#                               the option exists so callers can opt in
#                               once the isolated branch merges. Toggling
#                               ON without HIP sources present is a no-op
#                               plus a status message.
# With both OFF, only the pure-CPU test binaries build. A configure will
# succeed but produce nothing that can run a model.
option(MIMIRMIND_ENABLE_L0  "Build the Level Zero backend and all L0-tied consumers" ON)
option(MIMIRMIND_ENABLE_HIP "Build the HIP backend (placeholder — no impl on main yet)" OFF)

# C++23 required — std::expected in src/core/ipc/ for the M-Munin
# error-return convention (see decisions/2026-07-13-m-munin-scope.md).
# GCC 13+ and Clang 16+ both ship it in <expected>. If a build host
# stalls on this, upgrading the toolchain is preferable to reverting
# the ipc layer to an ad-hoc Result<T,E> substitute.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
)

# ---- Level Zero loader + SPIR-V toolchain -----------------------------------
#
# Both blocks only run when the L0 backend is enabled. FATAL_ERROR remains
# fatal in that case — an explicit opt-in must find its dependencies. With
# MIMIRMIND_ENABLE_L0=OFF nothing here executes, no L0 headers or ocloc
# need to be installed on the host.

if(MIMIRMIND_ENABLE_L0)
    find_path(LEVEL_ZERO_INCLUDE_DIR ze_api.h
        PATHS /usr/include /usr/local/include
        PATH_SUFFIXES level_zero
        DOC "Path to <level_zero/ze_api.h>"
    )

    find_library(LEVEL_ZERO_LIBRARY
        NAMES ze_loader
        PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
        DOC "Path to libze_loader.so"
    )

    if(NOT LEVEL_ZERO_INCLUDE_DIR OR NOT LEVEL_ZERO_LIBRARY)
        message(FATAL_ERROR
            "\n  Level Zero loader not found (MIMIRMIND_ENABLE_L0=ON requested it).\n"
            "  On Debian/Ubuntu, install with:\n"
            "    sudo apt install level-zero level-zero-dev\n"
            "  For recent versions, use Intel's APT repo:\n"
            "    https://dgpu-docs.intel.com/driver/client/overview.html\n"
            "  Or build the loader from source (github.com/oneapi-src/level-zero) into /usr/local.\n"
            "  Configure with -DMIMIRMIND_ENABLE_L0=OFF to skip the Level Zero build entirely.\n"
        )
    endif()

    message(STATUS "Level Zero headers: ${LEVEL_ZERO_INCLUDE_DIR}")
    message(STATUS "Level Zero library: ${LEVEL_ZERO_LIBRARY}")

    find_program(OCLOC_EXECUTABLE
        NAMES ocloc
        DOC "Intel OpenCL Offline Compiler — compiles .cl to .spv")

    if(NOT OCLOC_EXECUTABLE)
        message(FATAL_ERROR
            "\n  ocloc not found (MIMIRMIND_ENABLE_L0=ON requires it for SPV kernel build).\n"
            "  On Debian/Ubuntu via Intel repo: apt install intel-ocloc libigc2 libigdfcl2\n"
            "  (or the Intel client driver bundle from\n"
            "   https://dgpu-docs.intel.com/driver/client/overview.html)\n"
            "  Configure with -DMIMIRMIND_ENABLE_L0=OFF to skip.\n")
    endif()

    message(STATUS "ocloc: ${OCLOC_EXECUTABLE}")
endif()

if(MIMIRMIND_ENABLE_HIP)
    # HIP CMake package ships with ROCm at /opt/rocm/lib/cmake/hip. Consumers
    # on non-default ROCm installs can override via -DHIP_DIR=... or by
    # exporting HIP_PATH. Once found, `hip::host` provides the include dirs
    # + libamdhip64 link line for host-side HIP runtime consumers.
    #
    # The skeleton only uses host-side HIP (hipGetDeviceCount, hipSetDevice,
    # hipGetDeviceProperties). Actual kernel compilation (setting
    # CMAKE_HIP_ARCHITECTURES + enabling the HIP language) comes with the
    # first kernel port on this branch, not with the skeleton.
    if(NOT DEFINED HIP_PATH AND NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm" CACHE PATH "Path to the ROCm/HIP install")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}")
    find_package(hip QUIET CONFIG)
    if(NOT hip_FOUND)
        message(FATAL_ERROR
            "\n  HIP not found (MIMIRMIND_ENABLE_HIP=ON requested it).\n"
            "  On Ubuntu with ROCm installed via amdgpu-install, HIP CMake\n"
            "  configs live at /opt/rocm/lib/cmake/hip. Override with\n"
            "    -DHIP_PATH=/path/to/rocm\n"
            "  or export HIP_PATH before configure.\n"
            "  Configure with -DMIMIRMIND_ENABLE_HIP=OFF to skip.\n")
    endif()
    message(STATUS "HIP found at: ${HIP_PATH} (version ${hip_VERSION})")

    # hipcc — for compiling .hip source into standalone hsaco code
    # objects that HipModule loads at runtime via hipModuleLoadData.
    # `--genco` is the hipcc mode that produces one .hsaco per input.
    find_program(HIPCC_EXECUTABLE
        NAMES hipcc
        HINTS "${HIP_PATH}/bin" /usr/bin /usr/local/bin
        DOC "hipcc — compiles .hip to .hsaco (with --genco)")
    if(NOT HIPCC_EXECUTABLE)
        message(FATAL_ERROR
            "\n  hipcc not found (MIMIRMIND_ENABLE_HIP=ON requires it for\n"
            "  .hip -> .hsaco kernel builds). Install via ROCm's\n"
            "  amdgpu-install --usecase=rocm, or add -DHIPCC_EXECUTABLE=<path>.\n")
    endif()
    message(STATUS "hipcc: ${HIPCC_EXECUTABLE}")

    # Target arch(s) for hipcc --genco. Default targets our home-lab
    # Radeon RX 7800 XT (Navi 32). Override with e.g.
    #   -DMIMIRMIND_HIP_ARCHS="gfx1101;gfx1100;gfx1102"
    # once the kernel port validates on multiple boards.
    set(MIMIRMIND_HIP_ARCHS "gfx1101"
        CACHE STRING "HIP offload archs for hipcc --genco (semicolon list)")
endif()

if(MIMIRMIND_ENABLE_L0)

set(KERNEL_SOURCES
    kernels/rmsnorm.cl
    kernels/matmul_q4k_vec.cl
    kernels/matmul_q5k_vec.cl
    kernels/matmul_q6k_vec.cl
    kernels/matmul_q8_0_vec.cl
    kernels/matmul_q6k_gemm.cl
    kernels/matmul_q4k_gemm.cl
    kernels/matmul_q8_0_gemm.cl
    kernels/qkv_split.cl
    kernels/add_bias.cl
    kernels/add_residual.cl
    kernels/silu_mul.cl
    kernels/rope_inplace.cl
    kernels/mul_scalar.cl
    kernels/gelu_mul.cl
    kernels/rmsnorm_gemma.cl
    kernels/rmsnorm_no_weight.cl
    kernels/rmsnorm_qkv.cl
    kernels/add_rmsnorm.cl
    kernels/rope_inplace_ff.cl
    kernels/attention.cl
    kernels/attention_flash_partial.cl
    kernels/attention_flash_merge.cl
    kernels/attention_prefill_flash.cl
    kernels/attention_fp16.cl
    kernels/attention_flash_partial_fp16.cl
    kernels/attention_prefill_flash_fp16.cl
    kernels/rmsnorm_qkv_fp16.cl
    kernels/qkv_split_fp16.cl
    kernels/rope_inplace_fp16.cl
    kernels/rope_inplace_ff_fp16.cl
    kernels/kv_quant_commit_q8_0.cl
    kernels/attention_q8_0.cl
    kernels/attention_flash_partial_q8_0.cl
    kernels/attention_prefill_flash_q8_0.cl
    kernels/attention_prefill_flash_q8_0_gqa.cl
    kernels/scaled_add_residual.cl
    kernels/x_quant_i8.cl
    kernels/matmul_q8_0_vec_dp4a.cl
    kernels/matmul_q4k_vec_dp4a.cl
    kernels/matmul_q8_0_gemm_v2.cl
    kernels/matmul_q6k_gemm_v2.cl
    kernels/matmul_q4k_gemm_v2.cl
    kernels/matmul_q8_0_vec_reorder.cl
    kernels/moe_down_fused_k_q6k.cl
    kernels/moe_down_fused_k_q8_0.cl
)

set(KERNEL_SPV_FILES "")
set(SPV_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/spv)
file(MAKE_DIRECTORY ${SPV_OUTPUT_DIR})

foreach(src IN LISTS KERNEL_SOURCES)
    get_filename_component(kname ${src} NAME_WE)
    set(spv ${SPV_OUTPUT_DIR}/${kname}.spv)
    add_custom_command(
        OUTPUT ${spv}
        # ocloc -spv_only writes <name>_<device>.spv by default; without
        # -device the suffix is just "_" so we'd get rmsnorm_.spv.
        # -output_no_suffix drops that, but it still uses the input file's
        # basename — exactly what we want.
        COMMAND ${OCLOC_EXECUTABLE} compile
                -file ${CMAKE_CURRENT_SOURCE_DIR}/${src}
                -options "-cl-std=CL3.0"
                -spv_only
                -output_no_suffix
                -out_dir ${SPV_OUTPUT_DIR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
        WORKING_DIRECTORY ${SPV_OUTPUT_DIR}
        COMMENT "ocloc -> ${kname}.spv"
        VERBATIM
    )
    list(APPEND KERNEL_SPV_FILES ${spv})
endforeach()

# Second build of attention_prefill_flash_q8_0_gqa.cl with a smaller
# K-tile. Same source, ocloc gets `-D ATTN_FLASH_PREFILL_KTILE=64` so
# the `#ifndef` in the kernel picks up the override. We copy the source
# to a distinct basename first because ocloc's -output_no_suffix names
# the SPV after the input basename — that lets both variants coexist
# in the same spv/ directory. Dispatch picks at runtime via
# `features.flashPrefillKTileQ8` in config.json.
set(GQA_KTILE64_CL  ${CMAKE_CURRENT_BINARY_DIR}/attention_prefill_flash_q8_0_gqa_ktile64.cl)
set(GQA_KTILE64_SPV ${SPV_OUTPUT_DIR}/attention_prefill_flash_q8_0_gqa_ktile64.spv)
add_custom_command(
    OUTPUT ${GQA_KTILE64_CL}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_prefill_flash_q8_0_gqa.cl
            ${GQA_KTILE64_CL}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_prefill_flash_q8_0_gqa.cl
    VERBATIM
)
add_custom_command(
    OUTPUT ${GQA_KTILE64_SPV}
    COMMAND ${OCLOC_EXECUTABLE} compile
            -file ${GQA_KTILE64_CL}
            -options "-cl-std=CL3.0 -D ATTN_FLASH_PREFILL_KTILE=64"
            -spv_only
            -output_no_suffix
            -out_dir ${SPV_OUTPUT_DIR}
    DEPENDS ${GQA_KTILE64_CL}
    WORKING_DIRECTORY ${SPV_OUTPUT_DIR}
    COMMENT "ocloc -> attention_prefill_flash_q8_0_gqa_ktile64.spv (KTILE=64)"
    VERBATIM
)
list(APPEND KERNEL_SPV_FILES ${GQA_KTILE64_SPV})

add_custom_target(spv_kernels ALL DEPENDS ${KERNEL_SPV_FILES})

endif() # MIMIRMIND_ENABLE_L0 (SPV kernel block)

# cpp-httplib pulls in <thread> and <pthread> for its server loop; also
# used by mimirmind_core_common::Log for its background file sink.
find_package(Threads REQUIRED)

# ---- mimirmind_core_common — pure-CPU foundation library --------------------
#
# The pieces that never touch Level Zero: config parsing, GGUF loading,
# logging, on-wire IPC serde (framing, TensorManifest, WireOps), MuninClient
# (pure serde against the wire), MappedFile (mmap wrapper), governor-lock
# helpers. Buildable on any host, no GPU SDK required. Pure-CPU unit tests
# link only this library.

add_library(mimirmind_core_common STATIC
    src/core/backend/BackendRegistry.cpp
    src/core/config/Config.cpp
    src/core/cpu/CpuContext.cpp
    src/core/gguf/GgufTypes.cpp
    src/core/gguf/TensorFingerprint.cpp
    src/core/gguf/WeightsMap.cpp
    src/core/ipc/TensorManifest.cpp
    src/core/ipc/UnixSocketFrame.cpp
    src/core/ipc/WireOps.cpp
    src/core/os/MappedFile.cpp
    src/core/log/Log.cpp
    src/core/os/GovernorLock.cpp
)

target_include_directories(mimirmind_core_common PUBLIC src)

# Vendored nlohmann/json (Config). SYSTEM so upstream code does not trip
# our strict warning flags in consumers.
target_include_directories(mimirmind_core_common SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(mimirmind_core_common PUBLIC
    Threads::Threads
)

# BackendRegistry::probeAll() forward-declares
# `mimirmind::core::l0::probeBackend()` and
# `mimirmind::core::hip::probeBackend()` behind these defines. Their
# implementations live in mimirmind_core_l0 / mimirmind_core_hip
# respectively — pure-preprocessor gating here, no header or link
# dep on the backend libs. PUBLIC so binaries that include
# BackendRegistry.hpp see the same picture as _common.cpp does.
if(MIMIRMIND_ENABLE_L0)
    target_compile_definitions(mimirmind_core_common PUBLIC MIMIRMIND_HAVE_L0=1)
endif()
if(MIMIRMIND_ENABLE_HIP)
    target_compile_definitions(mimirmind_core_common PUBLIC MIMIRMIND_HAVE_HIP=1)
endif()

# ---- mimirmind_core_l0 — Level Zero-tied foundation ------------------------
#
# Everything that touches Level Zero handles directly: L0Context, USM
# allocation, IPC handle export/import, MuninClient's L0 wrapper surface
# (transitively — the header includes L0Context.hpp), ChunkAllocator (same).
# Links against libze_loader and PUBLIC-links mimirmind_core_common so
# consumers only need one target to depend on.
#
# Only built when MIMIRMIND_ENABLE_L0=ON.

if(MIMIRMIND_ENABLE_L0)
    add_library(mimirmind_core_l0 STATIC
        src/core/gpu/l0/L0Context.cpp
        src/core/gpu/l0/L0ComputeContext.cpp
        src/core/gpu/l0/UsmAllocator.cpp
        src/core/gpu/l0/CommandQueue.cpp
        src/core/gpu/l0/GpuKernel.cpp
        src/core/gpu/l0/GpuModule.cpp
        src/core/ipc/IpcExporter.cpp
        src/core/ipc/IpcImporter.cpp
        src/core/ipc/MuninClient.cpp
        src/core/gguf/GgufReader.cpp
        src/munin/ChunkAllocator.cpp
    )

    target_include_directories(mimirmind_core_l0 PUBLIC
        ${LEVEL_ZERO_INCLUDE_DIR}
    )

    target_link_libraries(mimirmind_core_l0 PUBLIC
        mimirmind_core_common
        ${LEVEL_ZERO_LIBRARY}
    )

    # Legacy alias so existing target_link_libraries(... mimirmind_core)
    # calls below keep working during the transition. Once every
    # executable is migrated to explicit _common / _l0, this alias can
    # be removed.
    add_library(mimirmind_core ALIAS mimirmind_core_l0)
endif()

# ---- mimirmind_core_hip — AMD HIP-tied foundation --------------------------
#
# Peer to `mimirmind_core_l0`: HipContext + (later, on this branch) HIP
# stream/module/allocator wrappers. PUBLIC-links `mimirmind_core_common`
# so consumers only need one target to depend on for the neutral +
# HIP-native surface. Kernels come as follow-up commits on the same
# feat/hip-backend-skeleton branch — the skeleton only carries the
# device-enumeration + feature-probe layer today.
#
# Only built when MIMIRMIND_ENABLE_HIP=ON.

if(MIMIRMIND_ENABLE_HIP)
    add_library(mimirmind_core_hip STATIC
        src/core/gpu/hip/HipContext.cpp
        src/core/gpu/hip/HipComputeContext.cpp
        src/core/gpu/hip/HipEvent.cpp
        src/core/gpu/hip/HipKernel.cpp
        src/core/gpu/hip/HipMemoryAllocator.cpp
        src/core/gpu/hip/HipModule.cpp
        src/core/gpu/hip/HipStream.cpp

        # Schritt 3b sub-A: HipGpuOps skeleton — implements the
        # backend-neutral ComputeOps interface for the HIP path.
        # All kernel-launch methods currently throw; feature-flag
        # accessors + Q8_0 reorder counters + module loading are
        # real. Follow-up commits fill launches group-by-group.
        src/compute/hip/GpuOps.cpp

        # Schritt 3b sub-F: HipGpuMatmul skeleton — the ComputeMatmul
        # peer to HipGpuOps. Loads the 5 Q8_0 matmul kernels
        # (vec / gemm / gemm_v2 / vec_dp4a / moe_down_fused_k).
        # supports/dp4aAvailable/moeDownFusedKAvailable/sync/
        # autotuneReport are real; matmul launches throw pending
        # follow-up fill-in commits.
        src/compute/hip/GpuMatmul.cpp
    )

    target_link_libraries(mimirmind_core_hip PUBLIC
        mimirmind_core_common
        hip::host
    )

    # Schicht 4 — when both L0 and HIP are compiled in, the L0 lib carries
    # InferenceEngine.cpp which #ifdefs into compute/hip/GpuOps.hpp and
    # core/gpu/hip/HipComputeContext.hpp for the factory dispatch. Pull
    # the HIP lib in as a link dep so the transitive `hip::host` include
    # paths and libs reach mimirmind_core_l0 and its dependents (the
    # mimirmind binary, gpu_tests, prefill_bench). No-op when L0=OFF.
    if(MIMIRMIND_ENABLE_L0)
        target_link_libraries(mimirmind_core_l0 PUBLIC mimirmind_core_hip)
    endif()

    # ---- HIP kernel code objects (.hip → .hsaco via hipcc --genco) ---
    #
    # Parallel to the ocloc-based L0 spv_kernels custom target. Each .hip
    # source lands as one .hsaco file in ${CMAKE_BINARY_DIR}/hsaco/
    # which HipModule loads at runtime. Multi-arch fat-binary support
    # ships automatically if MIMIRMIND_HIP_ARCHS lists more than one arch.
    set(HIP_KERNEL_SOURCES
        kernels_hip/hip_probe_kernel.hip
        kernels_hip/add_bias.hip
        kernels_hip/add_residual.hip
        kernels_hip/add_rmsnorm.hip
        kernels_hip/attention.hip
        kernels_hip/attention_flash_merge.hip
        kernels_hip/attention_flash_partial.hip
        kernels_hip/attention_flash_partial_fp16.hip
        kernels_hip/attention_flash_partial_q8_0.hip
        kernels_hip/attention_fp16.hip
        kernels_hip/attention_prefill_flash.hip
        kernels_hip/attention_prefill_flash_fp16.hip
        kernels_hip/attention_prefill_flash_q8_0.hip
        kernels_hip/attention_prefill_flash_q8_0_gqa.hip
        kernels_hip/attention_q8_0.hip
        kernels_hip/kv_quant_commit_q8_0.hip
        kernels_hip/matmul_q8_0_gemm.hip
        kernels_hip/matmul_q8_0_gemm_v2.hip
        kernels_hip/matmul_q4k_vec.hip
        kernels_hip/matmul_q5_0_vec.hip
        kernels_hip/matmul_q5k_vec.hip
        kernels_hip/matmul_q6k_vec.hip
        kernels_hip/matmul_q8_0_vec.hip
        kernels_hip/matmul_q8_0_vec_dp4a.hip
        kernels_hip/matmul_q8_0_vec_reorder.hip
        kernels_hip/moe_down_fused_k_q8_0.hip
        kernels_hip/mul_scalar.hip
        kernels_hip/qkv_split.hip
        kernels_hip/qkv_split_fp16.hip
        kernels_hip/rmsnorm.hip
        kernels_hip/rmsnorm_gemma.hip
        kernels_hip/rmsnorm_no_weight.hip
        kernels_hip/rmsnorm_qkv.hip
        kernels_hip/rmsnorm_qkv_fp16.hip
        kernels_hip/rope_inplace.hip
        kernels_hip/rope_inplace_fp16.hip
        kernels_hip/rope_inplace_ff.hip
        kernels_hip/scaled_add_residual.hip
        kernels_hip/gelu_mul.hip
        kernels_hip/x_quant_i8.hip
        kernels_hip/silu_mul.hip
    )

    set(HSACO_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/hsaco)
    file(MAKE_DIRECTORY ${HSACO_OUTPUT_DIR})

    # Turn MIMIRMIND_HIP_ARCHS ("gfx1101" or "gfx1101;gfx1100") into the
    # hipcc CLI form "--offload-arch=gfx1101 --offload-arch=gfx1100".
    set(HIP_OFFLOAD_ARGS "")
    foreach(arch IN LISTS MIMIRMIND_HIP_ARCHS)
        list(APPEND HIP_OFFLOAD_ARGS "--offload-arch=${arch}")
    endforeach()

    set(HIP_HSACO_FILES "")
    foreach(src IN LISTS HIP_KERNEL_SOURCES)
        get_filename_component(kname ${src} NAME_WE)
        set(hsaco ${HSACO_OUTPUT_DIR}/${kname}.hsaco)
        add_custom_command(
            OUTPUT ${hsaco}
            COMMAND ${HIPCC_EXECUTABLE}
                    --genco
                    ${HIP_OFFLOAD_ARGS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${src}
                    -o ${hsaco}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
            WORKING_DIRECTORY ${HSACO_OUTPUT_DIR}
            COMMENT "hipcc --genco -> ${kname}.hsaco"
            VERBATIM
        )
        list(APPEND HIP_HSACO_FILES ${hsaco})
    endforeach()

    # Second build of attention_prefill_flash_q8_0_gqa.hip with a
    # smaller K-tile. Same source, hipcc gets
    # `-DATTN_FLASH_PREFILL_KTILE=64` so the `#ifndef` in the kernel
    # picks up the override. Copy the source to a distinct basename
    # first because hipcc names the code object after the input
    # filename — that lets both variants coexist in the same hsaco/
    # dir. Kernel symbol stays `attention_prefill_flash_q8_0_gqa`
    # (unchanged by the macro) so HipModule::getKernel finds it under
    # the same name from either .hsaco. Dispatch picks between the two
    # at runtime via `features.flashPrefillKTileQ8` in config.json.
    # Mirrors the L0 ocloc `-D ATTN_FLASH_PREFILL_KTILE=64` block above.
    set(HIP_GQA_KTILE64_HIP
        ${CMAKE_CURRENT_BINARY_DIR}/attention_prefill_flash_q8_0_gqa_ktile64.hip)
    set(HIP_GQA_KTILE64_HSACO
        ${HSACO_OUTPUT_DIR}/attention_prefill_flash_q8_0_gqa_ktile64.hsaco)
    add_custom_command(
        OUTPUT ${HIP_GQA_KTILE64_HIP}
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/kernels_hip/attention_prefill_flash_q8_0_gqa.hip
                ${HIP_GQA_KTILE64_HIP}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels_hip/attention_prefill_flash_q8_0_gqa.hip
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${HIP_GQA_KTILE64_HSACO}
        COMMAND ${HIPCC_EXECUTABLE}
                --genco
                ${HIP_OFFLOAD_ARGS}
                -DATTN_FLASH_PREFILL_KTILE=64
                ${HIP_GQA_KTILE64_HIP}
                -o ${HIP_GQA_KTILE64_HSACO}
        DEPENDS ${HIP_GQA_KTILE64_HIP}
        WORKING_DIRECTORY ${HSACO_OUTPUT_DIR}
        COMMENT "hipcc --genco -> attention_prefill_flash_q8_0_gqa_ktile64.hsaco (KTILE=64)"
        VERBATIM
    )
    list(APPEND HIP_HSACO_FILES ${HIP_GQA_KTILE64_HSACO})

    add_custom_target(hip_kernels ALL DEPENDS ${HIP_HSACO_FILES})
endif()

# ---- L0-backed targets: mimirmind binary, munin daemon, GPU tests ---------
#
# All of these consume L0Context / GpuOps / GpuMatmul / CommandQueue directly
# and cannot link without libze_loader. Gated on MIMIRMIND_ENABLE_L0.

if(MIMIRMIND_ENABLE_L0)

add_executable(mimirmind
    src/mimirmind/main.cpp
    src/mimirmind/CliParser.cpp
    src/mimirmind/ParityMode.cpp
    src/mimirmind/ServeMode.cpp
    src/mimirmind/SmokeMode.cpp
    src/mimirmind/diagnostics/Formatting.cpp
    src/mimirmind/diagnostics/SmokeSuite.cpp
    src/compute/Activations.cpp
    src/compute/Attention.cpp
    src/compute/Dequant.cpp
    src/compute/Embedding.cpp
    src/compute/cpu/GpuMatmul.cpp
    src/compute/cpu/GpuOps.cpp
    src/compute/l0/GpuMatmul.cpp
    src/compute/l0/GpuOps.cpp
    src/compute/Matmul.cpp
    src/compute/MoeRouting.cpp
    src/compute/Norm.cpp
    src/compute/QuantTypeRegistry.cpp
    src/compute/Rope.cpp
    src/compute/Sampling.cpp
    src/compute/Softmax.cpp
    src/compute/SpeculativeSampler.cpp
    src/compute/quant/Bfloat16.cpp
    src/compute/quant/Float16.cpp
    src/compute/quant/Float32.cpp
    src/compute/quant/Q4K.cpp
    src/compute/quant/Q5K.cpp
    src/compute/quant/Q5_0.cpp
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
    src/model/ChatTemplate.cpp
    src/model/FusedQkvWeights.cpp
    src/model/LlmConfig.cpp
    src/model/ResponseCleaner.cpp
    src/model/Tokenizer.cpp
    src/runtime/BlockBuffers.cpp
    src/runtime/InferenceEngine.cpp
    src/runtime/KvCache.cpp
    src/runtime/SpeculativeDecoder.cpp
    src/runtime/FanController.cpp
    src/runtime/GpuClockGovernor.cpp
    src/runtime/OpProfiler.cpp
    src/runtime/PerfRegressionDetector.cpp
    src/runtime/PowerMonitor.cpp
    src/runtime/SystemMonitor.cpp
    src/runtime/ThermalGuard.cpp
    src/runtime/ThermalProfile.cpp
    src/runtime/arch/ArchBackend.cpp
    src/runtime/arch/GemmaBaseBackend.cpp
    src/runtime/arch/Gemma4Backend.cpp
    src/runtime/arch/Gemma4DenseBackend.cpp
    src/runtime/arch/Gemma4E4BBackend.cpp
    src/runtime/arch/Gemma4MoeBackend.cpp
    src/runtime/arch/Qwen2Backend.cpp
    src/server/ApiHelpers.cpp
    src/server/ApiServer.cpp
    src/server/ChatCompletionHandler.cpp
    src/server/ChatRequestParser.cpp
    src/server/PromptTrimmer.cpp
    src/server/RequestDispatcher.cpp
    src/server/RequestTracker.cpp
    src/server/SseEncoder.cpp
    src/server/SystemStatusBuilder.cpp
)

add_dependencies(mimirmind spv_kernels)

# cpp-httplib is only used by the server, not by mimirmind_core. SYSTEM so
# upstream code does not trip our strict warning flags.
target_include_directories(mimirmind SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib
)

target_link_libraries(mimirmind PRIVATE
    mimirmind_core
)

# ---- munin — persistent model-memory daemon (M-Munin Schritt 7) -------------
#
# The Munin daemon holds N models in USM(host) and exports them over an
# AF_UNIX socket to attached mimirmind workers via L0 IPC (SCM_RIGHTS +
# zeMemOpenIpcHandle). Standalone binary; links against mimirmind_core
# only. No SPV kernels, no httplib, no compute code — see the M-Munin
# scope ADR (decisions/2026-07-13-m-munin-scope.md) for the tier-1
# boundary.
#
# Run via:
#   docker compose run --rm munin --config /etc/mimirmind/config.json

add_executable(munin
    src/munin/main.cpp
    src/munin/AttachSession.cpp
    src/munin/Daemon.cpp
    src/munin/ModelStore.cpp
    src/munin/SocketServer.cpp
)

target_link_libraries(munin PRIVATE mimirmind_core)

endif() # MIMIRMIND_ENABLE_L0 (mimirmind + munin executables)

# ---- Pure-CPU test binaries — always built ---------------------------------
#
# These need no L0, so they build on any host (Ryzen/Apple/ARM). They link
# only mimirmind_core_common. On hosts without L0 they are the only smoke
# signal that the build+toolchain is healthy.

# ---- quant_tests — pure-CPU unit tests for compute::QuantType ---------------
#
# Standalone test binary. No Level Zero, no SPV kernels — exercises the
# dequant logic via hand-crafted byte blocks. Useful to verify the
# refactor preserves behaviour and to catch regressions in new quant
# types without needing the iGPU.
#
# Run via:   cmake --build build --target quant_tests && build/quant_tests

add_executable(quant_tests
    tests/quant_tests.cpp
    src/compute/Dequant.cpp
    src/compute/QuantTypeRegistry.cpp
    src/compute/quant/Bfloat16.cpp
    src/compute/quant/Float16.cpp
    src/compute/quant/Float32.cpp
    src/compute/quant/Q4K.cpp
    src/compute/quant/Q5K.cpp
    src/compute/quant/Q5_0.cpp
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
)

target_include_directories(quant_tests PRIVATE tests)

target_link_libraries(quant_tests PRIVATE mimirmind_core_common)

# ---- arch_tests — pure-CPU tests for arch-related logic ---------------------
#
# Covers the testable surface of the architecture backends without GPU:
#   - isSupportedArchitecture()  string predicate
#   - compute::softmaxRows       row-wise stable softmax + causal mask
#   - compute::multiHeadAttention  GQA causal MHA (pure CPU, double-acc)
#
# Building runBlock() of the concrete backends would need real USM + SPV
# modules — that stays an integration test, not in this binary.
#
# Run via:   cmake --build build --target arch_tests && build/arch_tests

add_executable(arch_tests
    tests/arch_tests.cpp
    src/compute/Attention.cpp
    src/compute/MoeRouting.cpp
    src/compute/Softmax.cpp
    src/model/ResponseCleaner.cpp
    src/runtime/GpuClockGovernor.cpp
    src/runtime/PowerMonitor.cpp
    src/runtime/SystemMonitor.cpp
    src/runtime/ThermalGuard.cpp
    src/runtime/ThermalProfile.cpp
)

target_include_directories(arch_tests PRIVATE tests)

target_link_libraries(arch_tests PRIVATE mimirmind_core_common)

# ---- ipc_tests — pure-CPU tests for src/core/ipc primitives ---------------
#
# Foundation for the M-Munin daemon (see decisions/2026-07-13-m-munin-scope.md):
# covers UnixSocketFrame length-prefix framing + SCM_RIGHTS fd passing over a
# socketpair, plus TensorManifest JSON serde round-trip and error paths. The
# L0-side IPC wrappers (IpcExporter / IpcImporter) live in mimirmind_core and
# are exercised end-to-end by the existing tools/l0-ipc-testrig on real
# hardware; there is no way to unit-test them without a live Level Zero
# device, so we don't try.
#
# Run via:   cmake --build build --target ipc_tests && build/ipc_tests

add_executable(ipc_tests
    tests/ipc_tests.cpp
)

target_include_directories(ipc_tests PRIVATE tests)

target_link_libraries(ipc_tests PRIVATE mimirmind_core_common)

# ---- munin_tests — Munin wire-op layer + ChunkAllocator layout math --------
#
# Covers RequestEnvelope, HealthzResponse, error-envelope helpers, and
# ChunkAllocator::layoutInsideChunk (arithmetic-only path, but the class
# lives in mimirmind_core_l0 because its .hpp includes L0Context). Gated
# on MIMIRMIND_ENABLE_L0 for that reason; the socket integration
# (AttachSession + SocketServer) still needs a live L0 device + peer
# worker and lives in the docker-based smoke tests from Schritt 8.
#
# Run via:   cmake --build build --target munin_tests && build/munin_tests

if(MIMIRMIND_ENABLE_L0)
    add_executable(munin_tests
        tests/munin_tests.cpp
    )
    target_include_directories(munin_tests PRIVATE tests)
    target_link_libraries(munin_tests PRIVATE mimirmind_core_l0)
endif()

# ---- compute_tests — pure-CPU tests for CPU compute primitives -------------
#
# Covers what's testable in compute/ without GPU or model file:
#   - Sampler              (greedy / top-k / top-p / seed determinism)
#   - matmul (CPU)         (F32 / Q8_0 / Q6_K hand-crafted blocks)
#   - embeddingLookup      (F32 / Q6_K / out-of-range zero-fill)
#
# Run via:   cmake --build build --target compute_tests && build/compute_tests

add_executable(compute_tests
    tests/compute_tests.cpp
    src/compute/Dequant.cpp
    src/compute/Embedding.cpp
    src/compute/Matmul.cpp
    src/compute/QuantTypeRegistry.cpp
    src/compute/Sampling.cpp
    src/compute/quant/Bfloat16.cpp
    src/compute/quant/Float16.cpp
    src/compute/quant/Float32.cpp
    src/compute/quant/Q4K.cpp
    src/compute/quant/Q5K.cpp
    src/compute/quant/Q5_0.cpp
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
)

target_include_directories(compute_tests PRIVATE tests)

target_link_libraries(compute_tests PRIVATE mimirmind_core_common)

# ---- L0-backed integration binaries ----------------------------------------
#
# gpu_tests / prefill_bench / l0_ipc_testrig all need libze_loader plus the
# compiled SPV kernels. Gated on MIMIRMIND_ENABLE_L0.

if(MIMIRMIND_ENABLE_L0)

# ---- gpu_tests — verify GPU kernels match CPU reference math ---------------
#
# Integration-level: requires an Intel iGPU reachable via /dev/dri plus the
# SPV kernels under /usr/local/share/mimirmind/spv. We're NOT testing the
# GPU hardware — we verify our SPV kernels compute the same values as the
# CPU references in compute::rmsNorm / siluInPlace / applyRopeInPlace /
# compute::matmul. Each test allocates USM, dispatches the kernel, and
# compares output element-wise with a tolerance.
#
# Run via:   docker compose run --rm mimirmind /usr/local/bin/gpu_tests

add_executable(gpu_tests
    tests/gpu_tests.cpp
    src/compute/Activations.cpp
    src/compute/Attention.cpp
    src/compute/Dequant.cpp
    src/compute/Embedding.cpp
    src/compute/cpu/GpuMatmul.cpp
    src/compute/cpu/GpuOps.cpp
    src/compute/l0/GpuMatmul.cpp
    src/compute/l0/GpuOps.cpp
    src/compute/Matmul.cpp
    src/compute/MoeRouting.cpp
    src/compute/Norm.cpp
    src/compute/QuantTypeRegistry.cpp
    src/compute/Rope.cpp
    src/compute/Sampling.cpp
    src/compute/Softmax.cpp
    src/compute/quant/Bfloat16.cpp
    src/compute/quant/Float16.cpp
    src/compute/quant/Float32.cpp
    src/compute/quant/Q4K.cpp
    src/compute/quant/Q5K.cpp
    src/compute/quant/Q5_0.cpp
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
)

add_dependencies(gpu_tests spv_kernels)

target_include_directories(gpu_tests PRIVATE tests)

target_link_libraries(gpu_tests PRIVATE mimirmind_core)

# ---- prefill_bench — wall-clock micro-benchmark for GpuOps::attentionAsync -
#
# Integration-level micro-benchmark for the flash-prefill path landed in
# M9.8b. Not a unit test: it measures wall-clock latency at a caller-
# configurable T_k and prints a ledger-ready Markdown row plus one NDJSON
# line for programmatic aggregation. Same L0/USM/GpuOps stack as gpu_tests
# so the build closure mirrors it.
#
# Run via:
#   docker compose run --rm mimirmind /usr/local/bin/prefill_bench --tk 24576
#   docker compose run --rm mimirmind /usr/local/bin/prefill_bench --tk 32768

add_executable(prefill_bench
    tests/prefill_bench.cpp
    src/compute/Activations.cpp
    src/compute/Attention.cpp
    src/compute/Dequant.cpp
    src/compute/Embedding.cpp
    src/compute/cpu/GpuMatmul.cpp
    src/compute/cpu/GpuOps.cpp
    src/compute/l0/GpuMatmul.cpp
    src/compute/l0/GpuOps.cpp
    src/compute/Matmul.cpp
    src/compute/MoeRouting.cpp
    src/compute/Norm.cpp
    src/compute/QuantTypeRegistry.cpp
    src/compute/Rope.cpp
    src/compute/Sampling.cpp
    src/compute/Softmax.cpp
    src/compute/quant/Bfloat16.cpp
    src/compute/quant/Float16.cpp
    src/compute/quant/Float32.cpp
    src/compute/quant/Q4K.cpp
    src/compute/quant/Q5K.cpp
    src/compute/quant/Q5_0.cpp
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
)

add_dependencies(prefill_bench spv_kernels)

target_include_directories(prefill_bench PRIVATE tests)

target_link_libraries(prefill_bench PRIVATE mimirmind_core)

# ---- l0_ipc_testrig — go/no-go gate for M-Munin ----------------------------
#
# Standalone diagnostic: two-process test of zeMemGetIpcHandle /
# zeMemOpenIpcHandle on the target iGPU. Independent of mimirmind_core
# (uses raw L0 API only) so a failure here isolates the question to
# Level Zero itself, not our wrapper.
#
# Not installed in the runtime image — this is a hardware-verification
# tool, not a shipped component. Build + run inside the builder
# container on the target host:
#
#   docker compose run --rm builder bash -lc '
#       cmake --build /src/build --target l0_ipc_testrig &&
#       /src/tools/l0-ipc-testrig.sh /src/build/l0_ipc_testrig
#   '
#
# The runner in tools/l0-ipc-testrig.sh launches owner + attacher and
# reports TESTRIG: PASS or TESTRIG: FAIL.

add_executable(l0_ipc_testrig
    tools/l0-ipc-testrig.cpp
)

target_include_directories(l0_ipc_testrig PRIVATE
    ${LEVEL_ZERO_INCLUDE_DIR}
)

target_link_libraries(l0_ipc_testrig PRIVATE
    ${LEVEL_ZERO_LIBRARY}
)

endif() # MIMIRMIND_ENABLE_L0 (gpu_tests + prefill_bench + l0_ipc_testrig)

# ---- hip_probe — HIP-side "toolchain is live end-to-end" diagnostic --------
#
# Enumerates all HIP-visible devices, instantiates HipContext on the
# auto-selected one, prints the portable BackendDeviceInfo + the HIP-
# native superset, and evaluates every BackendFeature flag. Companion to
# l0_ipc_testrig for the AMD side.
#
# Not installed in any runtime image — this is a bringup diagnostic, not
# a shipped component. Build + run once the HIP backend skeleton lands:
#
#   cmake --build build --target hip_probe && ./build/hip_probe

if(MIMIRMIND_ENABLE_HIP)
    add_executable(hip_probe
        tools/hip-probe.cpp
    )

    target_link_libraries(hip_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_mem_probe — round-trip sanity for HipMemoryAllocator -----
    #
    # Alloc + H2D + D2H + byte-compare per HipAllocKind. Companion to
    # hip_probe: hip_probe proves the runtime + toolchain, this proves
    # the memory-layer + RAII wrapper. Not installed in any runtime image.

    add_executable(hip_mem_probe
        tools/hip-mem-probe.cpp
    )

    target_link_libraries(hip_mem_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_module_probe — end-to-end HipModule + HipKernel sanity ---
    #
    # Loads the hip_probe_kernel.hsaco produced by hipcc --genco, resolves
    # the write_pattern symbol, launches on a HipStream with HipEvent
    # timing, and byte-compares the device output against the host
    # reference. First tool that exercises the full HIP path: context +
    # allocator + stream + event + module + kernel + launch.

    add_executable(hip_module_probe
        tools/hip-module-probe.cpp
    )

    add_dependencies(hip_module_probe hip_kernels)

    target_link_libraries(hip_module_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_rmsnorm_probe — first real HIP kernel bit-parity test ----
    #
    # Loads rmsnorm.hsaco (compiled from kernels_hip/rmsnorm.hip via
    # hipcc --genco), runs RMSNorm on synthetic M×K input, compares
    # against an inline double-precision CPU reference within abs/rel
    # 1e-4. First concrete "GPU output matches expected" signal from
    # the HIP backend — foundation for the parity harness the rest of
    # the kernel-set will be validated against.

    add_executable(hip_rmsnorm_probe
        tools/hip-rmsnorm-probe.cpp
    )

    add_dependencies(hip_rmsnorm_probe hip_kernels)

    target_link_libraries(hip_rmsnorm_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_matmul_q8_probe — Q8_0 dequant + matmul bit-parity -------
    #
    # Second real HIP kernel. Loads matmul_q8_0_vec.hsaco, runs it on a
    # random Q8_0-packed weight matrix and a random X vector, compares
    # against a double-precision CPU reference that dequantizes the
    # same weights + does the dot product. First tool that exercises
    # the on-the-fly dequantization path — the bit-parity signal here
    # covers both the fp16 scale load and the int8 quant multiply.

    add_executable(hip_matmul_q8_probe
        tools/hip-matmul-q8-probe.cpp
    )

    add_dependencies(hip_matmul_q8_probe hip_kernels)

    target_link_libraries(hip_matmul_q8_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_rope_probe — RoPE (rotary embedding) parity check --------
    #
    # Third kernel port: element-wise, no LDS, but the first that
    # exercises the "runtime int slot via device pointer" launch-arg
    # pattern (startPosPtr) which the L0 side needs for CLR replay and
    # the HIP side will need for the same reason once we bring up
    # command-list-replay-equivalent (hipGraph capture) on RDNA3.

    add_executable(hip_rope_probe
        tools/hip-rope-probe.cpp
    )

    add_dependencies(hip_rope_probe hip_kernels)

    target_link_libraries(hip_rope_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_silu_mul_probe — SwiGLU fused step parity check ----------
    #
    # Trivial element-wise kernel (one thread per element, no LDS, no
    # reduction), but it's the FFN backbone in Llama / Qwen / Gemma —
    # every layer runs it. Uses the combined-tolerance pattern from
    # rope_probe (|diff| <= abs_tol + rel_tol * |ref|) since silu
    # passes through zero for negative inputs.

    add_executable(hip_silu_mul_probe
        tools/hip-silu-mul-probe.cpp
    )

    add_dependencies(hip_silu_mul_probe hip_kernels)

    target_link_libraries(hip_silu_mul_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_add_residual_probe — in-place residual add parity check --
    #
    # Trivial 1D element-wise (y += x). Every transformer block calls
    # this twice (post-attn, post-ffn). Combined-tolerance with tight
    # bounds — a single add should have essentially zero drift.

    add_executable(hip_add_residual_probe
        tools/hip-add-residual-probe.cpp
    )

    add_dependencies(hip_add_residual_probe hip_kernels)

    target_link_libraries(hip_add_residual_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_add_bias_probe — broadcast bias add parity check ---------
    #
    # Flat-1D thread index across (M, K), inner modulo picks bias slot.
    # Tight combined-tolerance; the interesting bit is that the index
    # math is exercised (modulo K per thread).

    add_executable(hip_add_bias_probe
        tools/hip-add-bias-probe.cpp
    )

    add_dependencies(hip_add_bias_probe hip_kernels)

    target_link_libraries(hip_add_bias_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_mul_scalar_probe — in-place scalar multiply parity ------
    #
    # First HIP probe to bind a plain-float POD via
    # HipKernel::setValue<float> — Gemma 4 layer_output_scale relies on
    # this by-value arg path. Otherwise trivially element-wise.

    add_executable(hip_mul_scalar_probe
        tools/hip-mul-scalar-probe.cpp
    )

    add_dependencies(hip_mul_scalar_probe hip_kernels)

    target_link_libraries(hip_mul_scalar_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_rmsnorm_gemma_probe — Gemma (1+weight) rmsnorm variant --
    #
    # Same tree reduction as vanilla rmsnorm, applies (1 + weight[k])
    # instead of weight[k]. Used for every proper Gemma 4 norm weight.

    add_executable(hip_rmsnorm_gemma_probe
        tools/hip-rmsnorm-gemma-probe.cpp
    )

    add_dependencies(hip_rmsnorm_gemma_probe hip_kernels)

    target_link_libraries(hip_rmsnorm_gemma_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_rmsnorm_no_weight_probe — Gemma V-norm rmsnorm variant --
    #
    # Same tree reduction, drops the multiplicative weight — Gemma 4's
    # V projection passes through a bare ggml_rms_norm before entering
    # the KV cache.

    add_executable(hip_rmsnorm_no_weight_probe
        tools/hip-rmsnorm-no-weight-probe.cpp
    )

    add_dependencies(hip_rmsnorm_no_weight_probe hip_kernels)

    target_link_libraries(hip_rmsnorm_no_weight_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_rmsnorm_qkv_probe — fused Q + K + V rmsnorm --------------
    #
    # One dispatch normalizes all three streams. Routes workgroups to
    # Q / K / V by grid index, writes K/V into their caches at the
    # curLen offset. Uses the RoPE-style device-int slot for curLen.
    # Probe additionally checks that pre-curLen cache slots are not
    # touched.

    add_executable(hip_rmsnorm_qkv_probe
        tools/hip-rmsnorm-qkv-probe.cpp
    )

    add_dependencies(hip_rmsnorm_qkv_probe hip_kernels)

    target_link_libraries(hip_rmsnorm_qkv_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_qkv_split_probe — fused-QKV output scatter ---------------
    #
    # First scatter/layout-transform kernel (no reduction, no LDS).
    # Splits a fused matmul output [M, Nq + 2*Nkv] into Q workspace +
    # K cache + V cache, K/V land at (curLen + m) * Nkv. Uses the
    # RoPE-style device-int slot for curLen. Scatter is a pure copy —
    # tolerance is 0 (bit-exact), any diff is a routing/offset bug.
    # Includes the cache-stomp check inherited from rmsnorm_qkv.

    add_executable(hip_qkv_split_probe
        tools/hip-qkv-split-probe.cpp
    )

    add_dependencies(hip_qkv_split_probe hip_kernels)

    target_link_libraries(hip_qkv_split_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_qkv_split_fp16_probe — fp16-KV variant of qkv_split ------
    #
    # Same scatter but K/V destinations are fp16 (__float2half on the
    # store). Q workspace stays fp32. Bit-exact against a CPU-computed
    # __float2half of the corresponding fused slice — both sides use
    # round-to-nearest-even.

    add_executable(hip_qkv_split_fp16_probe
        tools/hip-qkv-split-fp16-probe.cpp
    )

    add_dependencies(hip_qkv_split_fp16_probe hip_kernels)

    target_link_libraries(hip_qkv_split_fp16_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_rmsnorm_qkv_fp16_probe — fp16-KV variant of fused qkv ----
    #
    # Q branch: fp32 identical to rmsnorm_qkv. K/V branches promote
    # fp16 cache reads to fp32, compute the reduction and scale in fp32,
    # then round-trip results back to fp16. Combined-tolerance 1e-4/1e-4
    # for Q, 1e-3/1e-3 for K/V (fp16 store granularity + fp32 drift).
    # Cache-stomp check on the fp16 pre-curLen region.

    add_executable(hip_rmsnorm_qkv_fp16_probe
        tools/hip-rmsnorm-qkv-fp16-probe.cpp
    )

    add_dependencies(hip_rmsnorm_qkv_fp16_probe hip_kernels)

    target_link_libraries(hip_rmsnorm_qkv_fp16_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_flash_partial_probe — FlashAttention K-tile ---
    #
    # First non-trivial reduction+softmax+GEMV composite kernel in the
    # port. 2D grid (nHeads, nKTiles), wgSize=16 (half-wave on gfx1101).
    # Warp16 butterfly reduction replaces the L0 sub_group_reduce_max
    # / _add. Two-case probe: full-attention (both tiles valid) + SWA
    # (tile 0 hits the -INFINITY neutral-emission path). Bit-exact
    # match required on the neutral slots; combined-tolerance 1e-4/1e-4
    # on the numeric slots.

    add_executable(hip_attention_flash_partial_probe
        tools/hip-attention-flash-partial-probe.cpp
    )

    add_dependencies(hip_attention_flash_partial_probe hip_kernels)

    target_link_libraries(hip_attention_flash_partial_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_flash_merge_probe — FlashAttention merge ------
    #
    # Companion to the partial probe. Feeds a CPU-computed partial
    # tensor into the merge kernel, compares against a MONOLITHIC
    # (no-tiling) attention fp64 reference. Validates the online-
    # softmax identity round-trip. Two-case: A full-attention,
    # B SWA=32 (tile 0 neutral — merge must handle beta=0 slots).

    add_executable(hip_attention_flash_merge_probe
        tools/hip-attention-flash-merge-probe.cpp
    )

    add_dependencies(hip_attention_flash_merge_probe hip_kernels)

    target_link_libraries(hip_attention_flash_merge_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_flash_partial_fp16_probe — fp16-KV partial ----
    #
    # fp16-KV variant of the flash partial kernel. K/V are __half; the
    # kernel promotes via __half2float per load. Output partials stay
    # fp32 so the merge kernel is reused unchanged. Same two-case probe
    # pattern as the f32 variant (full-attention + SWA-with-neutral-
    # emission).

    add_executable(hip_attention_flash_partial_fp16_probe
        tools/hip-attention-flash-partial-fp16-probe.cpp
    )

    add_dependencies(hip_attention_flash_partial_fp16_probe hip_kernels)

    target_link_libraries(hip_attention_flash_partial_fp16_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_probe — untiled M5f.3 variant (a) --------------
    #
    # Non-flash attention kernel. 2D grid (nHeads, T_q), 64 KiB static
    # LDS for scores[MAX_TK=16384]. Two-case probe (full-causal + SWA)
    # exercises the causal-offset math across multiple query positions.

    add_executable(hip_attention_probe
        tools/hip-attention-probe.cpp
    )

    add_dependencies(hip_attention_probe hip_kernels)

    target_link_libraries(hip_attention_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_fp16_probe — fp16-KV untiled attention --------
    #
    # Same as hip_attention_probe but with fp16-typed K and V, promoted
    # via __half2float per load. Q and OUT stay fp32.

    add_executable(hip_attention_fp16_probe
        tools/hip-attention-fp16-probe.cpp
    )

    add_dependencies(hip_attention_fp16_probe hip_kernels)

    target_link_libraries(hip_attention_fp16_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_kv_quant_commit_q8_0_probe — fp32 → Q8_0 KV writer -----
    #
    # First write-side quantization kernel. Byte-exact vs a CPU Q8_0
    # quantiser that follows the same absmax + __float2half + roundf-
    # to-int8 recipe. Runs at wgSize=32 (full wave on gfx1101) — no
    # width scoping. Includes cache-stomp check on pre-curLen bytes.

    add_executable(hip_kv_quant_commit_q8_0_probe
        tools/hip-kv-quant-commit-q8-0-probe.cpp
    )

    add_dependencies(hip_kv_quant_commit_q8_0_probe hip_kernels)

    target_link_libraries(hip_kv_quant_commit_q8_0_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_q8_0_probe — Q8_0-KV untiled attention -------
    #
    # Same as hip_attention_probe but K/V are ggml-style Q8_0 blocks
    # (2 B fp16 scale + 32 B int8 quants). CPU reference dequantises
    # the exact same bytes the kernel does.

    add_executable(hip_attention_q8_0_probe
        tools/hip-attention-q8-0-probe.cpp
    )

    add_dependencies(hip_attention_q8_0_probe hip_kernels)

    target_link_libraries(hip_attention_q8_0_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_flash_partial_q8_0_probe — Q8_0 flash-partial -
    #
    # Q8_0-KV variant of the flash partial kernel. Partials stay fp32
    # so the merge kernel is reused unchanged for the Q8_0 KV path.

    add_executable(hip_attention_flash_partial_q8_0_probe
        tools/hip-attention-flash-partial-q8-0-probe.cpp
    )

    add_dependencies(hip_attention_flash_partial_q8_0_probe hip_kernels)

    target_link_libraries(hip_attention_flash_partial_q8_0_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_prefill_flash_probe — streaming prefill (f32) --
    #
    # Single-workgroup streaming online-softmax for T_q > 1. Small SLM
    # (~2.5 KiB) so occupancy is unconstrained. 2-case probe (full
    # causal + SWA) vs monolithic CPU attention.

    add_executable(hip_attention_prefill_flash_probe
        tools/hip-attention-prefill-flash-probe.cpp
    )

    add_dependencies(hip_attention_prefill_flash_probe hip_kernels)

    target_link_libraries(hip_attention_prefill_flash_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_prefill_flash_fp16_probe — fp16-KV prefill ----

    add_executable(hip_attention_prefill_flash_fp16_probe
        tools/hip-attention-prefill-flash-fp16-probe.cpp
    )

    add_dependencies(hip_attention_prefill_flash_fp16_probe hip_kernels)

    target_link_libraries(hip_attention_prefill_flash_fp16_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_prefill_flash_q8_0_probe — Q8_0-KV prefill ----

    add_executable(hip_attention_prefill_flash_q8_0_probe
        tools/hip-attention-prefill-flash-q8-0-probe.cpp
    )

    add_dependencies(hip_attention_prefill_flash_q8_0_probe hip_kernels)

    target_link_libraries(hip_attention_prefill_flash_q8_0_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_attention_prefill_flash_q8_0_gqa_probe — GQA-packed ----
    #
    # Colfax-style query-head packing over KV-groups. Grid changes to
    # (nKvHeads, T_q, 1); output layout unchanged. Per-Q-head state
    # arrays with #pragma unroll bounded by N_Q_PER_KV_MAX. K/V blocks
    # loaded once per tile step, reused across every Q-head in the
    # group — GQA-ratio KV-bandwidth reduction.

    add_executable(hip_attention_prefill_flash_q8_0_gqa_probe
        tools/hip-attention-prefill-flash-q8-0-gqa-probe.cpp
    )

    add_dependencies(hip_attention_prefill_flash_q8_0_gqa_probe hip_kernels)

    target_link_libraries(hip_attention_prefill_flash_q8_0_gqa_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_matmul_q8_gemm_probe — M-way GEMM with Q8_0 weights -----
    #
    # M-way generalisation of matmul_q8_0_vec. First multi-sub-group
    # kernel in the port: WgSize=64 with 4 explicit 16-lane sub-groups
    # via tid/16 and tid%16, warp16_reduce_sum over `__shfl_xor(v, off,
    # width=16)`. SLM 32 KiB (X-tile 1024 × M_TILE 8).

    add_executable(hip_matmul_q8_gemm_probe
        tools/hip-matmul-q8-gemm-probe.cpp
    )

    add_dependencies(hip_matmul_q8_gemm_probe hip_kernels)

    target_link_libraries(hip_matmul_q8_gemm_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_matmul_q8_gemm_v2_probe — 4× smaller SLM revision ------
    #
    # Identical semantics, X-tile shrunk from 1024 → 256 (SLM 32 → 8
    # KiB). Same parity check with different load-pattern; useful for
    # bench-comparison.

    add_executable(hip_matmul_q8_gemm_v2_probe
        tools/hip-matmul-q8-gemm-v2-probe.cpp
    )

    add_dependencies(hip_matmul_q8_gemm_v2_probe hip_kernels)

    target_link_libraries(hip_matmul_q8_gemm_v2_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_matmul_q8_vec_reorder_probe — layout-transformed matvec -
    #
    # Reorder splits each row into scales-first + quants-later. Total
    # row size unchanged (nBlocks * 34 B). The probe validates that the
    # reorder is correctness-neutral: same math, different load layout.

    add_executable(hip_matmul_q8_vec_reorder_probe
        tools/hip-matmul-q8-vec-reorder-probe.cpp
    )

    add_dependencies(hip_matmul_q8_vec_reorder_probe hip_kernels)

    target_link_libraries(hip_matmul_q8_vec_reorder_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_matmul_q8_vec_dp4a_probe — DP4A matvec, first intrinsic -
    #
    # FIRST use of a HIP compiler intrinsic in the port:
    # __builtin_amdgcn_sdot4(a, b, c, false) mapped to gfx1101 v_dot4.
    # Pre-quantised int8 X + Q8_0 W, one int32 dot per 4-element chunk,
    # single fp32 multiply per block for d × xScale.

    add_executable(hip_matmul_q8_vec_dp4a_probe
        tools/hip-matmul-q8-vec-dp4a-probe.cpp
    )

    add_dependencies(hip_matmul_q8_vec_dp4a_probe hip_kernels)

    target_link_libraries(hip_matmul_q8_vec_dp4a_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_moe_down_fused_k_q8_0_probe — MoE down-projection ------
    #
    # Fused over k active experts. First read-modify-write kernel in
    # the port — accum starts seeded and gets summed into. Probe pre-
    # seeds accum with random content and verifies GPU output equals
    # seed + computed_delta.

    add_executable(hip_moe_down_fused_k_q8_0_probe
        tools/hip-moe-down-fused-k-q8-0-probe.cpp
    )

    add_dependencies(hip_moe_down_fused_k_q8_0_probe hip_kernels)

    target_link_libraries(hip_moe_down_fused_k_q8_0_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_gelu_mul_probe — GELU-tanh SwiGLU parity check -----------
    #
    # Same shape as silu_mul, GELU-tanh activation instead of SiLU.
    # Used by Gemma 3 / Gemma 4 FFN paths (LLM_FFN_GELU parallel form).
    # Combined-tolerance; tanh drift keeps rel-tol at 1e-5 vs 1e-6 for
    # the plain SiLU probe.

    add_executable(hip_gelu_mul_probe
        tools/hip-gelu-mul-probe.cpp
    )

    add_dependencies(hip_gelu_mul_probe hip_kernels)

    target_link_libraries(hip_gelu_mul_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_scaled_add_residual_probe — MoE fused-accumulate check ---
    #
    # Trivial 1D fmaf: dst[i] += scale * src[i]. Replaces the L0 kernel
    # of the same name; used by the Gemma 4 MoE per-expert loop to
    # collapse mulScalar + addResidual into one launch per expert.
    # Tight combined-tolerance (scale ~ 0.2, both operands in [-1, 1]).

    add_executable(hip_scaled_add_residual_probe
        tools/hip-scaled-add-residual-probe.cpp
    )

    add_dependencies(hip_scaled_add_residual_probe hip_kernels)

    target_link_libraries(hip_scaled_add_residual_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_x_quant_i8_probe — per-row int8 quant parity check --------
    #
    # Same shape as rmsnorm: one WG per row, tree reduction over amax,
    # per-thread strided quantise pass. Feeds the DP4A Q8_0 matvec path.
    # Byte-exact quant compare (0 slack) — `roundf` matches ggml's
    # quantize_row_q8_0_ref. Row M-1 is all-zeros to exercise the amax=0
    # sentinel that stores scale=0 + all-zero quants.

    add_executable(hip_x_quant_i8_probe
        tools/hip-x-quant-i8-probe.cpp
    )

    add_dependencies(hip_x_quant_i8_probe hip_kernels)

    target_link_libraries(hip_x_quant_i8_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_add_rmsnorm_probe — fused residual + RMSNorm parity check
    #
    # Same reduction shape as rmsnorm. Verifies both outputs the fused
    # kernel writes: the in-place-updated `x` (post-add) and the
    # normalised `y`. Two separate combined-tolerance envelopes since
    # the `x` update is a plain sum (tight) while `y` rides the RMS
    # reduction + rsqrt (looser, like base rmsnorm).

    add_executable(hip_add_rmsnorm_probe
        tools/hip-add-rmsnorm-probe.cpp
    )

    add_dependencies(hip_add_rmsnorm_probe hip_kernels)

    target_link_libraries(hip_add_rmsnorm_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_rope_fp16_probe — fp16-KV RoPE parity check --------------
    #
    # Kernel body promotes each fp16 pair to fp32 registers, rotates,
    # demotes back on store. Reference does the same fp16 round-trip
    # so drift is entirely from sin/cos + multiply-subtract + the two
    # fp16 stores. Tolerance widened (5e-4 abs / 5e-3 rel) vs the fp32
    # rope probe to cover ~2^-11 fp16 quantisation.

    add_executable(hip_rope_fp16_probe
        tools/hip-rope-fp16-probe.cpp
    )

    add_dependencies(hip_rope_fp16_probe hip_kernels)

    target_link_libraries(hip_rope_fp16_probe PRIVATE
        mimirmind_core_hip
    )

    # ---- hip_rope_ff_probe — freq-factors RoPE parity check -----------
    #
    # Adds a [halfDim] `freq_factors` buffer that divides the per-pair
    # theta. Same numerics envelope as the base rope probe since the
    # only change is one extra multiply on the frequency computation.

    add_executable(hip_rope_ff_probe
        tools/hip-rope-ff-probe.cpp
    )

    add_dependencies(hip_rope_ff_probe hip_kernels)

    target_link_libraries(hip_rope_ff_probe PRIVATE
        mimirmind_core_hip
    )
endif()

# ---- backend_probe — neutral backend enumeration diagnostic ---------------
#
# Prints one line per BackendKind mimirmind knows about, whether it was
# compiled in, whether a device is visible at runtime, and a short detail
# string. Always built (needs no backend sources beyond the common
# BackendRegistry itself) — but only sees `available=yes` for backends
# that were also compiled in. When both L0 and HIP are on it links to
# both concrete backend libs so their `probeBackend()` symbols resolve.

add_executable(backend_probe
    tools/backend-probe.cpp
)

target_link_libraries(backend_probe PRIVATE
    mimirmind_core_common
)

if(TARGET mimirmind_core_l0)
    target_link_libraries(backend_probe PRIVATE mimirmind_core_l0)
endif()

if(TARGET mimirmind_core_hip)
    target_link_libraries(backend_probe PRIVATE mimirmind_core_hip)
endif()

# ---- backend_select_probe — runtime backend-selection demo --------------
#
# End-to-end: reads MIMIRMIND_BACKEND (falling back to LevelZero),
# resolves via BackendRegistry::resolveKind(), constructs the concrete
# ComputeContext via BackendRegistry::createContext(), and prints the
# device info + feature flags observed through the neutral interface.
#
# Same link surface as backend_probe — links both concrete backend libs
# when they're compiled in so createContext() can reach the free-fn
# factories via the file-scope forward decls in BackendRegistry.cpp.

add_executable(backend_select_probe
    tools/backend-select-probe.cpp
)

target_link_libraries(backend_select_probe PRIVATE
    mimirmind_core_common
)

if(TARGET mimirmind_core_l0)
    target_link_libraries(backend_select_probe PRIVATE mimirmind_core_l0)
endif()

if(TARGET mimirmind_core_hip)
    target_link_libraries(backend_select_probe PRIVATE mimirmind_core_hip)
endif()
