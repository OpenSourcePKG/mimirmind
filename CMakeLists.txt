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

# ---- Targets ----------------------------------------------------------------

add_executable(mimirmind
    src/main.cpp
    src/runtime/L0Context.cpp
    src/runtime/Log.cpp
    src/runtime/UsmAllocator.cpp
)

target_include_directories(mimirmind PRIVATE
    src
    ${LEVEL_ZERO_INCLUDE_DIR}
)

target_link_libraries(mimirmind PRIVATE
    ${LEVEL_ZERO_LIBRARY}
)