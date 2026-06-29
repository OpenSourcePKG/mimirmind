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
    kernels/add_bias.cl
    kernels/add_residual.cl
    kernels/silu_mul.cl
    kernels/rope_inplace.cl
    kernels/mul_scalar.cl
    kernels/gelu_mul.cl
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
    src/compute/Norm.cpp
    src/compute/Rope.cpp
    src/compute/Sampling.cpp
    src/compute/Softmax.cpp
    src/model/ChatTemplate.cpp
    src/model/GgufReader.cpp
    src/model/GgufTypes.cpp
    src/model/LlmConfig.cpp
    src/model/Tokenizer.cpp
    src/model/WeightsMap.cpp
    src/runtime/CommandQueue.cpp
    src/runtime/GpuKernel.cpp
    src/runtime/GpuModule.cpp
    src/runtime/InferenceEngine.cpp
    src/runtime/KvCache.cpp
    src/runtime/L0Context.cpp
    src/runtime/Log.cpp
    src/runtime/MappedFile.cpp
    src/runtime/UsmAllocator.cpp
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