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
)

target_include_directories(vk_ppu_icd PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${_vulkan_inc}
)

target_link_libraries(vk_ppu_icd PRIVATE ${CMAKE_DL_LIBS})

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

# --- manifest --------------------------------------------------------------
# Written next to the build output so VK_ICD_FILENAMES can point straight at it.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/ppu_icd.json.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ppu_icd.json"
  @ONLY
)
