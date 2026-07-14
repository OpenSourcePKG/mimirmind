cmake_minimum_required(VERSION 3.21)
project(mimirmind
    VERSION 0.0.1
    DESCRIPTION "Standalone C++ inference engine for GGUF on Intel iGPU via Level Zero"
    LANGUAGES CXX
)

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

# ---- Level Zero loader ------------------------------------------------------

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
        "\n  Level Zero loader not found.\n"
        "  On Debian/Ubuntu, install with:\n"
        "    sudo apt install level-zero level-zero-dev\n"
        "  For recent versions, use Intel's APT repo:\n"
        "    https://dgpu-docs.intel.com/driver/client/overview.html\n"
        "  Looked for headers in /usr/include/level_zero and library ze_loader in standard paths.\n"
    )
endif()

message(STATUS "Level Zero headers: ${LEVEL_ZERO_INCLUDE_DIR}")
message(STATUS "Level Zero library: ${LEVEL_ZERO_LIBRARY}")

# ---- SPIR-V kernel compilation (ocloc) --------------------------------------

find_program(OCLOC_EXECUTABLE
    NAMES ocloc
    DOC "Intel OpenCL Offline Compiler — compiles .cl to .spv")

if(NOT OCLOC_EXECUTABLE)
    message(FATAL_ERROR
        "\n  ocloc not found.\n"
        "  On Debian/Ubuntu via Intel repo: apt install intel-ocloc\n"
        "  (or the Intel client driver bundle from\n"
        "   https://dgpu-docs.intel.com/driver/client/overview.html)\n")
endif()

message(STATUS "ocloc: ${OCLOC_EXECUTABLE}")

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

# cpp-httplib pulls in <thread> and <pthread> for its server loop; also
# used by mimirmind_core::Log for its background file sink.
find_package(Threads REQUIRED)

# ---- mimirmind_core — shared static library --------------------------------
#
# Foundation code shared between mimirmind (the inference engine) and future
# munin (persistent model-memory daemon, see M-Munin ADR). Contains only the
# pieces both binaries must touch: Level-Zero context, USM allocation, GGUF
# parsing, config loading, logging. No compute kernels, no HTTP, no
# inference logic.

add_library(mimirmind_core STATIC
    src/core/config/Config.cpp
    src/core/gguf/GgufReader.cpp
    src/core/gguf/GgufTypes.cpp
    src/core/gguf/TensorFingerprint.cpp
    src/core/gguf/WeightsMap.cpp
    src/core/ipc/IpcExporter.cpp
    src/core/ipc/IpcImporter.cpp
    src/core/ipc/MuninClient.cpp
    src/core/ipc/TensorManifest.cpp
    src/core/ipc/UnixSocketFrame.cpp
    src/core/ipc/WireOps.cpp
    src/core/l0/L0Context.cpp
    src/core/l0/MappedFile.cpp
    src/core/l0/UsmAllocator.cpp
    src/core/log/Log.cpp
    src/core/os/GovernorLock.cpp
    src/munin/ChunkAllocator.cpp
)

# PUBLIC so consumers get the transitive header include paths (ze_api.h
# through L0Context.hpp, nlohmann/json through Config.hpp).
target_include_directories(mimirmind_core PUBLIC
    src
    ${LEVEL_ZERO_INCLUDE_DIR}
)

# Vendored nlohmann/json (Config). SYSTEM so upstream code does not trip
# our strict warning flags in consumers of mimirmind_core.
target_include_directories(mimirmind_core SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(mimirmind_core PUBLIC
    ${LEVEL_ZERO_LIBRARY}
    Threads::Threads
)

# ---- Targets ----------------------------------------------------------------

add_executable(mimirmind
    src/mimirmind/main.cpp
    src/mimirmind/CliParser.cpp
    src/mimirmind/diagnostics/Formatting.cpp
    src/mimirmind/diagnostics/SmokeSuite.cpp
    src/compute/Activations.cpp
    src/compute/Attention.cpp
    src/compute/Dequant.cpp
    src/compute/Embedding.cpp
    src/compute/GpuMatmul.cpp
    src/compute/GpuOps.cpp
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
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
    src/model/ChatTemplate.cpp
    src/model/FusedQkvWeights.cpp
    src/model/LlmConfig.cpp
    src/model/ResponseCleaner.cpp
    src/model/Tokenizer.cpp
    src/runtime/BlockBuffers.cpp
    src/runtime/CommandQueue.cpp
    src/runtime/GpuKernel.cpp
    src/runtime/GpuModule.cpp
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
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
)

target_include_directories(quant_tests PRIVATE tests)

target_link_libraries(quant_tests PRIVATE mimirmind_core)

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

target_link_libraries(arch_tests PRIVATE mimirmind_core)

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

target_link_libraries(ipc_tests PRIVATE mimirmind_core)

# ---- munin_tests — pure-CPU tests for Munin's wire-op layer ----------------
#
# Covers RequestEnvelope, HealthzResponse, and the error-envelope helpers.
# The socket integration (AttachSession + SocketServer) needs a live L0
# device + peer worker; that lives in the docker-based smoke tests that
# ship with Schritt 8, not here.
#
# Run via:   cmake --build build --target munin_tests && build/munin_tests

add_executable(munin_tests
    tests/munin_tests.cpp
)

target_include_directories(munin_tests PRIVATE tests)

target_link_libraries(munin_tests PRIVATE mimirmind_core)

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
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
)

target_include_directories(compute_tests PRIVATE tests)

target_link_libraries(compute_tests PRIVATE mimirmind_core)

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
    src/compute/GpuMatmul.cpp
    src/compute/GpuOps.cpp
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
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
    src/runtime/CommandQueue.cpp
    src/runtime/GpuKernel.cpp
    src/runtime/GpuModule.cpp
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
    src/compute/GpuMatmul.cpp
    src/compute/GpuOps.cpp
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
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
    src/runtime/CommandQueue.cpp
    src/runtime/GpuKernel.cpp
    src/runtime/GpuModule.cpp
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