cmake_minimum_required(VERSION 3.16)
project(vk_ppu_icd LANGUAGES CXX)

# Standalone, out-of-tree build. This directory is deliberately NOT part of the
# enclosing repository's CMake project.
#
#   cmake -S tools/vulkan-icd-ppu -B tools/vulkan-icd-ppu/build
#   cmake --build tools/vulkan-icd-ppu/build -j

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# --- Vulkan headers --------------------------------------------------------
# Prefer a vendored copy so the ICD builds on hosts (and on gong) that have no
# Vulkan development package installed.
set(_vulkan_inc "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/vulkan.h")
  set(_vulkan_inc "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
  message(STATUS "vk_ppu_icd: using vendored Vulkan headers")
else()
  find_package(Vulkan QUIET)
  if(Vulkan_INCLUDE_DIR)
    set(_vulkan_inc "${Vulkan_INCLUDE_DIR}")
    message(STATUS "vk_ppu_icd: using Vulkan headers at ${Vulkan_INCLUDE_DIR}")
  else()
    foreach(_cand /usr/include /usr/local/include)
      if(EXISTS "${_cand}/vulkan/vulkan.h")
        set(_vulkan_inc "${_cand}")
      endif()
    endforeach()
  endif()
  if(NOT _vulkan_inc)
    message(FATAL_ERROR
      "Vulkan headers not found. Install libvulkan-dev, or vendor them under "
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/.")
  endif()
endif()

add_library(vk_ppu_icd SHARED
  src/runtime.cpp
  src/instance.cpp
  src/device.cpp
  src/pipeline.cpp
  src/command.cpp
  src/submit.cpp
  src/query.cpp
  src/dispatch_table.cpp
  src/spirv_compile.cpp
  spirv-emit/emit.cpp
)

# SPIRV-Tools is the emitter's only dependency. Ubuntu/Debian ship it as a
# static-only package, so find the header directory and the archive by hand;
# linking it privately into the ICD keeps the ICD self-contained (an
# application must not need a SPIRV-Tools .so on its library path).
find_path(PPU_SPIRV_TOOLS_INCLUDE_DIR
  NAMES spirv-tools/libspirv.h
  HINTS ${SPIRV_TOOLS_ROOT} /usr/include /usr/local/include)
find_library(PPU_SPIRV_TOOLS_LIBRARY
  NAMES SPIRV-Tools libSPIRV-Tools.a
  HINTS ${SPIRV_TOOLS_ROOT} /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT PPU_SPIRV_TOOLS_INCLUDE_DIR OR NOT PPU_SPIRV_TOOLS_LIBRARY)
  message(FATAL_ERROR
    "SPIRV-Tools not found (need spirv-tools/libspirv.h and libSPIRV-Tools.a). "
    "Install libspirv-tools-dev, or set SPIRV_TOOLS_ROOT to a prefix that has both.")
endif()
message(STATUS "vk_ppu_icd: SPIRV-Tools at ${PPU_SPIRV_TOOLS_LIBRARY}")

target_include_directories(vk_ppu_icd PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PPU_SPIRV_TOOLS_INCLUDE_DIR}
  ${_vulkan_inc}
)

target_link_libraries(vk_ppu_icd PRIVATE
  ${CMAKE_DL_LIBS}
  ${PPU_SPIRV_TOOLS_LIBRARY}
)

# The ICD is loaded by the Vulkan loader; nothing links against it directly.
#
# Note: we deliberately do NOT compile with VK_NO_PROTOTYPES. The Vulkan
# headers' prototypes are the type checker for every entry point we define -
# omitting VK_NO_PROTOTYPES turned an argument-list mistake into a silently
# misinterpreted handle, so the compiler now enforces the signatures.
set_target_properties(vk_ppu_icd PROPERTIES OUTPUT_NAME "vk_ppu_icd")

target_compile_options(vk_ppu_icd PRIVATE -Wall -Wextra -Wno-unused-parameter)

# --- tests -----------------------------------------------------------------
add_executable(ppu_roundtrip tests/roundtrip.cpp)
target_include_directories(ppu_roundtrip PRIVATE ${_vulkan_inc})
target_link_libraries(ppu_roundtrip PRIVATE ${CMAKE_DL_LIBS})

# SPIR-V dispatch probe: compiles a .spv through the ICD's own
# vkCreateComputePipelines and checks the numbers against a CPU reference.
add_executable(ppu_spirv_dispatch tests/spirv_dispatch.cpp)
target_include_directories(ppu_spirv_dispatch PRIVATE ${_vulkan_inc})
target_link_libraries(ppu_spirv_dispatch PRIVATE ${CMAKE_DL_LIBS})

# Numeric probe: dispatch any kernel whose bindings are out/in/in2 + an 8-byte
# push constant, feed it exact inputs from files, and print the raw outputs so
# a lowering can be diffed against a host reference. No reference formula is
# baked in -- that is what lets it catch a runtime-only miscompile (a constant
# argument can be folded correctly by llc while the emitted instruction is
# wrong, which is exactly how the cos()/sin() ppu lowering bug stayed hidden).
add_executable(ppu_numprobe tests/numprobe.cpp)
target_include_directories(ppu_numprobe PRIVATE ${_vulkan_inc})
target_link_libraries(ppu_numprobe PRIVATE ${CMAKE_DL_LIBS})
# The emitter as a standalone tool, so the coverage sweeps in the other
# workstream (and on a host without the SDK) still have one binary to drive.
add_executable(ppu_emit spirv-emit/emit_main.cpp spirv-emit/emit.cpp)
target_include_directories(ppu_emit PRIVATE ${PPU_SPIRV_TOOLS_INCLUDE_DIR})
target_link_libraries(ppu_emit PRIVATE ${PPU_SPIRV_TOOLS_LIBRARY})

# --- manifest --------------------------------------------------------------
# Written next to the build output so VK_ICD_FILENAMES can point straight at it.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/ppu_icd.json.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ppu_icd.json"
  @ONLY
)
