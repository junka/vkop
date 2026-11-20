# Building the emitter standalone (for the coverage sweeps and for the other
# workstream) or linking it into the ICD is done by the parent CMakeLists;
# this file keeps the emitter buildable on its own too:
#
#   cmake -S . -B build && cmake --build build
#
# The only dependency is SPIRV-Tools (headers + the static library, which is
# what Debian/Ubuntu ship).
cmake_minimum_required(VERSION 3.16)
project(ppu_spirv_emit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Reused by the ICD's CMakeLists: SPIRV-Tools is a static-only package on
# Ubuntu, so this finds the header directory and the archive by hand.
find_path(SPIRV_TOOLS_INCLUDE_DIR
  NAMES spirv-tools/libspirv.h
  HINTS ${SPIRV_TOOLS_ROOT} /usr/include /usr/local/include)
find_library(SPIRV_TOOLS_LIBRARY
  NAMES SPIRV-Tools libSPIRV-Tools.a
  HINTS ${SPIRV_TOOLS_ROOT} /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT SPIRV_TOOLS_INCLUDE_DIR OR NOT SPIRV_TOOLS_LIBRARY)
  message(FATAL_ERROR
    "SPIRV-Tools not found (need spirv-tools/libspirv.h and libSPIRV-Tools.a). "
    "Install libspirv-tools-dev, or set SPIRV_TOOLS_ROOT.")
endif()

add_library(ppu_spirv_emit STATIC emit.cpp)
target_include_directories(ppu_spirv_emit PUBLIC ${SPIRV_TOOLS_INCLUDE_DIR})
target_link_libraries(ppu_spirv_emit PUBLIC ${SPIRV_TOOLS_LIBRARY})
set_target_properties(ppu_spirv_emit PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(emit emit_main.cpp)
target_link_libraries(emit PRIVATE ppu_spirv_emit)
