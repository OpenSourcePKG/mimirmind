# Minimal build of the xgrammar C++ core (vendored, MIT) for MimirMind's
# grammar-constrained tool decoding (roadmap 8.19.13.2). Python bindings,
# cpptrace stack traces and gtest are all disabled, so the only dependencies
# are the header-only picojson + dlpack bundled here. NOT llama.cpp/ggml.
cmake_minimum_required(VERSION 3.21)

file(GLOB_RECURSE XGRAMMAR_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cc")

add_library(xgrammar STATIC ${XGRAMMAR_CORE_SOURCES})

target_include_directories(xgrammar PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(xgrammar PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_include_directories(xgrammar SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include)
target_include_directories(xgrammar SYSTEM PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/picojson)

target_compile_definitions(xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)
target_compile_features(xgrammar PUBLIC cxx_std_17)
set_target_properties(xgrammar PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The vendored sources are third-party; do not fail our -Werror build on them.
if(NOT MSVC)
  target_compile_options(xgrammar PRIVATE -w)
endif()
