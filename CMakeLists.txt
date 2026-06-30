cmake_minimum_required(VERSION 3.21)
project(mimirmind
    VERSION 0.0.1
    DESCRIPTION "Standalone C++ inference engine for GGUF on Intel iGPU via Level Zero"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
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
    kernels/matmul_q6k_vec.cl
    kernels/matmul_q8_0_vec.cl
    kernels/add_bias.cl
    kernels/add_residual.cl
    kernels/silu_mul.cl
    kernels/rope_inplace.cl
    kernels/mul_scalar.cl
    kernels/gelu_mul.cl
    kernels/rmsnorm_gemma.cl
    kernels/rmsnorm_no_weight.cl
    kernels/rope_inplace_ff.cl
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

add_custom_target(spv_kernels ALL DEPENDS ${KERNEL_SPV_FILES})

# ---- Targets ----------------------------------------------------------------

add_executable(mimirmind
    src/main.cpp
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
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
    src/model/ChatTemplate.cpp
    src/model/GgufReader.cpp
    src/model/GgufTypes.cpp
    src/model/LlmConfig.cpp
    src/model/ResponseCleaner.cpp
    src/model/Tokenizer.cpp
    src/model/WeightsMap.cpp
    src/runtime/BlockBuffers.cpp
    src/runtime/CommandQueue.cpp
    src/runtime/GpuKernel.cpp
    src/runtime/GpuModule.cpp
    src/runtime/InferenceEngine.cpp
    src/runtime/KvCache.cpp
    src/runtime/L0Context.cpp
    src/runtime/Log.cpp
    src/runtime/GpuClockGovernor.cpp
    src/runtime/MappedFile.cpp
    src/runtime/PowerMonitor.cpp
    src/runtime/SystemMonitor.cpp
    src/runtime/ThermalGuard.cpp
    src/runtime/ThermalProfile.cpp
    src/runtime/UsmAllocator.cpp
    src/runtime/arch/ArchBackend.cpp
    src/runtime/arch/Gemma4Backend.cpp
    src/runtime/arch/Qwen2Backend.cpp
    src/server/ApiServer.cpp
)

add_dependencies(mimirmind spv_kernels)

target_include_directories(mimirmind PRIVATE
    src
    ${LEVEL_ZERO_INCLUDE_DIR}
)

# Vendored single-header libraries (cpp-httplib, nlohmann/json). SYSTEM
# so upstream code does not trip our strict warning flags.
target_include_directories(mimirmind SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# cpp-httplib pulls in <thread> and <pthread> for its server loop.
find_package(Threads REQUIRED)

target_link_libraries(mimirmind PRIVATE
    ${LEVEL_ZERO_LIBRARY}
    Threads::Threads
)

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
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
    src/model/GgufTypes.cpp
    src/runtime/Log.cpp
)

target_include_directories(quant_tests PRIVATE src tests)

target_link_libraries(quant_tests PRIVATE Threads::Threads)

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
    src/runtime/Log.cpp
    src/runtime/PowerMonitor.cpp
    src/runtime/SystemMonitor.cpp
    src/runtime/ThermalGuard.cpp
    src/runtime/ThermalProfile.cpp
)

target_include_directories(arch_tests PRIVATE src tests)

# ThermalProfile pulls in nlohmann/json — same vendored header the main
# target uses. SYSTEM so the test binary stays strict on our own code
# without dragging upstream warnings into our -Wall -Wextra -Wpedantic.
target_include_directories(arch_tests SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(arch_tests PRIVATE Threads::Threads)

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
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
    src/model/GgufTypes.cpp
    src/runtime/Log.cpp
)

target_include_directories(compute_tests PRIVATE src tests)

target_link_libraries(compute_tests PRIVATE Threads::Threads)

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
    src/compute/quant/Q6K.cpp
    src/compute/quant/Q8_0.cpp
    src/model/GgufTypes.cpp
    src/runtime/CommandQueue.cpp
    src/runtime/GpuKernel.cpp
    src/runtime/GpuModule.cpp
    src/runtime/L0Context.cpp
    src/runtime/Log.cpp
    src/runtime/UsmAllocator.cpp
)

add_dependencies(gpu_tests spv_kernels)

target_include_directories(gpu_tests PRIVATE
    src
    tests
    ${LEVEL_ZERO_INCLUDE_DIR}
)

target_link_libraries(gpu_tests PRIVATE
    ${LEVEL_ZERO_LIBRARY}
    Threads::Threads
)