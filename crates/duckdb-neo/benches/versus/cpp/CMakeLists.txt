# C++ runner of the versus benchmarks. `scripts/neo_versus.py build` configures it with
#   DUCKDB_CPP_DIR  duckdb_cpp.hpp and duckdb_cpp.cpp, extracted from a DuckDB checkout
#   DUCKDB_LIB_DIR  the libduckdb.so and duckdb_v2.h that libduckdb-sys downloaded for the Rust runner
#   VERSUS_LTO      ON to compile and link with -flto
cmake_minimum_required(VERSION 3.21)
project(neo_versus_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(VERSUS_LTO "Compile and link with -flto" OFF)

foreach(variable DUCKDB_CPP_DIR DUCKDB_LIB_DIR)
  if(NOT ${variable})
    message(FATAL_ERROR "${variable} is required")
  endif()
endforeach()

# -g only adds symbols for perf; it does not change code generation.
add_compile_options(-g)
if(VERSUS_LTO)
  add_compile_options(-flto)
  add_link_options(-flto -fuse-ld=lld)
endif()

find_library(
  DUCKDB_LIBRARY
  NAMES duckdb
  PATHS "${DUCKDB_LIB_DIR}"
  NO_DEFAULT_PATH NO_CACHE)
if(NOT DUCKDB_LIBRARY)
  message(FATAL_ERROR "no libduckdb in ${DUCKDB_LIB_DIR}")
endif()

# Built like DuckDB's own tools/cpp/CMakeLists.txt builds the C++ API.
add_library(duckdb_cpp_api STATIC "${DUCKDB_CPP_DIR}/duckdb_cpp.cpp")
target_include_directories(duckdb_cpp_api PUBLIC "${DUCKDB_CPP_DIR}" "${DUCKDB_LIB_DIR}")
target_link_libraries(duckdb_cpp_api PUBLIC "${DUCKDB_LIBRARY}")
set_target_properties(duckdb_cpp_api PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

# Every family file is a source of the executable itself, never of a library, so the linker keeps the static
# registrars in it. New family files are picked up at the next build.
file(GLOB VERSUS_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(versus_cpp ${VERSUS_SOURCES})
target_link_libraries(versus_cpp PRIVATE duckdb_cpp_api)
set_target_properties(versus_cpp PROPERTIES BUILD_RPATH "${DUCKDB_LIB_DIR}")
