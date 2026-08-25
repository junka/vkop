cmake_minimum_required(VERSION 3.10)

project(vkop)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_CLANG_TIDY clang-tidy)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE ccache)
if(CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE})
endif(CCACHE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(FP16_CXX_FLAGS "-mf16c")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(FP16_CXX_FLAGS "-march=armv8.2-a+fp16")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM")
    set(FP16_CXX_FLAGS "-mfp16-format=ieee")
else()
    message(WARNING "FP16 not supported on this architecture")
endif()

# ---- GCC + macOS libc++ ABI bridge -------------------------------------------
# PyTorch's macOS wheels are built against Apple's libc++ (std::__1 namespace).
# Homebrew GCC links against libstdc++ (std:: namespace), producing an ABI
# mismatch when objects compiled with GCC are linked against libtorch.
# Force GCC to use libc++ headers + libc++ runtime so all code shares the
# same C++ ABI as libtorch.
#
# Additionally, GCC's standalone -mf16c flag requires a matching -march and
# fails with "target specific option mismatch" when combined with the
# libc++ bridge.  Replace it with -march=native which implicitly enables
# FP16 on modern CPUs.
# -----------------------------------------------------------------------------
if(APPLE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
          AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpfullversion -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    message(STATUS "GCC detected on macOS (version ${GCC_VERSION})")

    if(GCC_VERSION VERSION_GREATER_EQUAL 15)
        message(FATAL_ERROR
            "GCC ${GCC_VERSION} + Apple libc++ is not supported because GCC 15 "
            "rejects the built-in traits (__decay, __add_pointer, ...) that "
            "modern Apple libc++ headers use in function signatures.\n"
            "\n"
            "Please choose one of these options:\n"
            "  1. Install GCC 14:    `brew install gcc@14` and rebuild with\n"
            "                        -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-14\n"
            "  2. Use Apple clang:   -DCMAKE_CXX_COMPILER=/usr/bin/clang++\n"
            "  3. Use GCC 13 or older: -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-13"
        )
    endif()

    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE APPLE_SDK_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT APPLE_SDK_ROOT)
        set(APPLE_SDK_ROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
    endif()

    set(LIBCXX_INCLUDE_DIR       "${APPLE_SDK_ROOT}/usr/include/c++/v1")
    set(LIBCXX_SYSTEM_INCLUDE_DIR "${APPLE_SDK_ROOT}/usr/include")
    set(LIBCXX_LIB_DIR           "${APPLE_SDK_ROOT}/usr/lib")

    message(STATUS "Bridging to Apple libc++ ABI (GCC ${GCC_VERSION})")
    message(STATUS "  libc++ headers: ${LIBCXX_INCLUDE_DIR}")
    message(STATUS "  libc++ lib dir: ${LIBCXX_LIB_DIR}")

    # Use CMAKE_CXX_FLAGS (not add_compile_options) so these flags only
    # apply to C++ files, not to C files (shaders/*.c).
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++ -isystem ${LIBCXX_INCLUDE_DIR} -isystem ${LIBCXX_SYSTEM_INCLUDE_DIR}")
    add_link_options(
        "SHELL:-nostdlib++"
        "SHELL:-L${LIBCXX_LIB_DIR}"
        "SHELL:-lc++"
    )

    # GCC's standalone -mf16c requires a matching -march.  Replace with
    # -march=native which implicitly enables FP16 support on modern CPUs.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(FP16_CXX_FLAGS "-march=native")
    endif()
endif()
if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od /Zi /W3 /fp:fast")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wall -Wextra -Werror -fno-omit-frame-pointer ${FP16_CXX_FLAGS}")
endif()

if(DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_PATH $ENV{VULKAN_SDK})
    message(STATUS "VULKAN_SDK env var found: ${VULKAN_SDK_PATH}")
    set(CMAKE_PREFIX_PATH ${VULKAN_SDK_PATH} ${CMAKE_PREFIX_PATH})
    # Use the VULKAN_SDK path to find Vulkan
    find_package(Vulkan QUIET PATHS ${VULKAN_SDK_PATH} NO_DEFAULT_PATH)
    if(NOT Vulkan_FOUND)
        message(STATUS "Vulkan config not found, using manual setup")
        if(EXISTS ${VULKAN_SDK_PATH}/include/vulkan/vulkan.h)
            set(Vulkan_INCLUDE_DIRS ${VULKAN_SDK_PATH}/include)
            set(Vulkan_FOUND TRUE)

            link_directories(${VULKAN_SDK_PATH}/lib)
        else()
            message(FATAL_ERROR "Cannot find Vulkan headers or libraries in ${VULKAN_SDK_PATH}")
        endif()
    endif()
else()
    if(DEFINED Vulkan_INCLUDE_DIRS)
        message(STATUS "Using custom Vulkan include directory: ${Vulkan_INCLUDE_DIRS}")
        set(Vulkan_FOUND TRUE)
        # Verify that the provided include path is valid
        if(NOT EXISTS ${Vulkan_INCLUDE_DIRS}/vulkan/vulkan.h)
            message(FATAL_ERROR "Vulkan header not found at ${Vulkan_INCLUDE_DIRS}/vulkan/vulkan.h")
        endif()
    else()
        # Fall back to system default paths
        find_package(Vulkan REQUIRED)
    endif()
endif()
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

# Source files
file(GLOB_RECURSE SRC_FILES vulkan/*.cpp ops/*.cpp core/*.cpp)
file(GLOB_RECURSE ALL_SOURCE_FILES vulkan/*.cpp vulkan/*.hpp ops/*.cpp core/*.cpp ops/*.hpp core/*.hpp)

if(DEFINED GLSLC_EXECUTABLE)
    if(EXISTS ${GLSLC_EXECUTABLE})
        message(STATUS "Using custom glslc executable: ${GLSLC_EXECUTABLE}")
    else()
        message(FATAL_ERROR "Custom glslc executable not found at: ${GLSLC_EXECUTABLE}")
    endif()
else()
    find_program(GLSLC_EXECUTABLE NAMES glslc PATHS ENV VULKAN_SDK PATH_SUFFIXES bin)    
    if(NOT GLSLC_EXECUTABLE)
        # Try common installation paths
        find_program(GLSLC_EXECUTABLE NAMES glslc HINTS
            /usr/bin
            /usr/local/bin
            $ENV{VULKAN_SDK}/bin
            /opt/vulkan-sdk/bin
        )
    endif()
endif()

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install Vulkan SDK or specify GLSLC_EXECUTABLE")
endif()

message(STATUS "Found glslc: ${GLSLC_EXECUTABLE}")

# Shader compilation
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(GLOB_RECURSE SHADER_FILES ${SHADER_DIR}/*.comp ${SHADER_DIR}/*.vert)
# Exclude shared #include-only headers in shaders/common/ from standalone
# compilation (they are pulled in by image/ and buffer/ shaders).
list(FILTER SHADER_FILES EXCLUDE REGEX "/common/")

# Ensure SPIRV output directory exists
file(MAKE_DIRECTORY ${SPIRV_DIR})

option(USE_FP16 "Enable usage of FP16" OFF)
if(USE_FP16)
    set(SHADER_FLAGS "-DFP16")
endif()

# spv_to_c.py: embeds a SPIR-V blob as `unsigned char <sym_name>_spv[]` +
# `<sym_name>_spv_len`. Used on all platforms (replaces the xxd+sed path,
# which derived the symbol from the xxd-of-filepath and could not disambiguate
# two shaders with the same basename in different folders).
set(PY_SCRIPT_CONTENT
"import sys
import os

def main():
    if len(sys.argv) != 4:
        print('Usage: spv_to_c.py <input.spv> <output.h> <symbol_name>')
        sys.exit(1)
    input_spv = sys.argv[1]
    output_h  = sys.argv[2]
    sym_name  = sys.argv[3] + '_spv'

    with open(input_spv, 'rb') as f:
        data = f.read()

    hex_lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        line = ', '.join(f'0x{b:02x}' for b in chunk)
        hex_lines.append(line)
    hex_str = ',\\n'.join(hex_lines)

    content = f'''// Auto-generated from {os.path.basename(input_spv)}
unsigned char {sym_name}[] = {{
{hex_str}
}};
unsigned int {sym_name}_len = {len(data)};
'''

    with open(output_h, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()
")
set(PY_SCRIPT_FILE "${CMAKE_BINARY_DIR}/spv_to_c.py")
file(WRITE "${PY_SCRIPT_FILE}" "${PY_SCRIPT_CONTENT}")
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Compile shaders to SPIR-V and embed as C headers. The embedded symbol is
# folder-derived: shaders/image/relu.comp -> image_relu_spv,
# shaders/buffer/unary.comp -> buffer_unary_spv. Both image and buffer
# shaders get their folder as a prefix, so same-named shaders in different
# folders never collide.
foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    file(RELATIVE_PATH SHADER_REL ${SHADER_DIR} ${SHADER})
    string(REPLACE "/" "_" SYM_BASE "${SHADER_REL}")
    get_filename_component(SYM_BASE "${SYM_BASE}" NAME_WE)
    # SPIRV and embed outputs are folder-prefixed so two same-named shaders
    # in different folders (image/concat vs buffer/concat) don't collide.
    set(SPIRV_FILE ${SPIRV_DIR}/${SYM_BASE}.spv)
    set(SPIRV_SOURCE_FILE ${SPIRV_DIR}/${SYM_BASE}.c)

    # Buffer shaders ship their own separate fp16 dual build (DUAL_FP16_SHADERS
    # below), so the base build here is always the fp32 variant — do NOT pass
    # -DFP16 to buffer shaders even when USE_FP16 is on (it would make the base
    # build an fp16 variant, colliding with the op's fp32-vs-fp16 selection).
    string(FIND "${SHADER_REL}" "buffer/" _is_buffer)
    if(_is_buffer EQUAL 0)
        set(_BUF_SHADER_FLAGS "")
    else()
        set(_BUF_SHADER_FLAGS "${SHADER_FLAGS}")
    endif()
    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER} -fshader-stage=compute -g --target-spv=spv1.3 -o ${SPIRV_FILE} ${_BUF_SHADER_FLAGS}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )
    add_custom_command(
        OUTPUT ${SPIRV_SOURCE_FILE}
        COMMAND ${Python3_EXECUTABLE}
            ${PY_SCRIPT_FILE}
            ${SPIRV_FILE}
            ${SPIRV_SOURCE_FILE}
            ${SYM_BASE}
        DEPENDS
            ${SPIRV_FILE}
            ${PY_SCRIPT_FILE}
        COMMENT "Embedding SPIR-V shader ${SYM_BASE} into C header"
    )
    set_source_files_properties(${SPIRV_SOURCE_FILE} PROPERTIES LANGUAGE C)
    list(APPEND SPIRV_HEADERS ${SPIRV_SOURCE_FILE})
    list(APPEND SRC_FILES ${SPIRV_SOURCE_FILE})
endforeach()

# Some shaders keep separate fp32/fp16 SPIR-V builds (no runtime branch in
# the shader; the operator picks the right one at construction). Each entry is
# a path relative to SHADER_DIR (e.g. buffer/gather, buffer/unary). The fp16
# build is compiled with -DFP16 and embedded as <folder>_<name>_fp16_spv
# (folder-derived, same as the base build).
#
# fp16 buffer storage packs two half values per uint word, so the fp16 path
# must use a word-per-thread store pattern to avoid read-modify-write races.
# Transpose is non-contiguous (permuted) so its fp16 path reads each output
# element via the packed-half BUF_LOAD and packs two results per output word.
set(DUAL_FP16_SHADERS
    buffer/gather
    buffer/sigmoid
    buffer/tanh
    buffer/atan
    buffer/softplus
    buffer/add
    buffer/sub
    buffer/mul
    buffer/div
    buffer/pow
    buffer/sqrt
    buffer/reduce
    buffer/prelu
    buffer/reshape
    buffer/expand
    buffer/transpose
    buffer/concat
    buffer/softmax
    buffer/matmul
    buffer/cos
    buffer/sin
    buffer/neg
)
foreach(DUAL_NAME ${DUAL_FP16_SHADERS})
    string(REPLACE "/" "_" DUAL_BASE "${DUAL_NAME}")
    set(SPIRV_FILE_FP16 ${SPIRV_DIR}/${DUAL_BASE}_fp16.spv)
    set(SPIRV_SOURCE_FILE_FP16 ${SPIRV_DIR}/${DUAL_BASE}_fp16.c)
    add_custom_command(
        OUTPUT ${SPIRV_FILE_FP16}
        COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/${DUAL_NAME}.comp
                -fshader-stage=compute -g --target-spv=spv1.3
                -o ${SPIRV_FILE_FP16} -DFP16
        DEPENDS ${SHADER_DIR}/${DUAL_NAME}.comp
        COMMENT "Compiling ${DUAL_NAME} (fp16) to SPIR-V"
    )
    add_custom_command(
        OUTPUT ${SPIRV_SOURCE_FILE_FP16}
        COMMAND ${Python3_EXECUTABLE}
            ${PY_SCRIPT_FILE}
            ${SPIRV_FILE_FP16}
            ${SPIRV_SOURCE_FILE_FP16}
            ${DUAL_BASE}_fp16
        DEPENDS
            ${SPIRV_FILE_FP16}
            ${PY_SCRIPT_FILE}
        COMMENT "Embedding fp16 SPIR-V shader ${DUAL_BASE}_fp16 into C header"
    )
    set_source_files_properties(${SPIRV_SOURCE_FILE_FP16} PROPERTIES LANGUAGE C)
    list(APPEND SPIRV_HEADERS ${SPIRV_SOURCE_FILE_FP16})
    list(APPEND SRC_FILES ${SPIRV_SOURCE_FILE_FP16})
endforeach()


# Add executable
add_library(vkop ${SRC_FILES})

# Add generated headers to the target
add_custom_target(compile_shaders DEPENDS ${SPIRV_HEADERS})
add_dependencies(vkop compile_shaders)

# Include SPIR-V headers
target_include_directories(vkop PRIVATE ${SPIRV_DIR})
# FlatBuffers: generated schema header + vendored runtime headers live under
# model/generated. SYSTEM so -Wall/-Wextra/-Werror skip third-party code.
target_include_directories(vkop SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/model/generated)
target_link_libraries(vkop PRIVATE ${CMAKE_DL_LIBS})

add_subdirectory(model)

find_program(CLANG_FORMAT "clang-format")
if(NOT CLANG_FORMAT)
  message(WARNING "clang-format not found, formatting step will be skipped.")
else()
  add_custom_target(
    format-code
    COMMAND ${CLANG_FORMAT} -style=file -i ${ALL_SOURCE_FILES}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Formatting all C++ source and header files..."
  )

  add_dependencies(vkop format-code)
endif()

find_program(CPPLINT "cpplint")
if(NOT CPPLINT)
    message(WARNING "cpplint not found, linting step will be skipped.")
else()
    add_custom_target(
        lint-code
        COMMAND ${CPPLINT} --extensions=cpp,hpp --counting=detailed ${ALL_SOURCE_FILES}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running cpplint on all C++ source and header files..."
    )
    # add_dependencies(vkop lint-code)
endif()

target_compile_definitions(vkop PUBLIC VK_ENABLE_BETA_EXTENSIONS)
target_compile_definitions(vkop PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

option(USE_VALIDATION_LAYERS "Enable validation layers" OFF)
option(USE_DEBUG_LAYERS "Enable debug layers" OFF)
option(USE_MEASURE_TIME "Enable measuring time" OFF)
option(USE_VMA "Enable usage of Vulkan Memory Allocator" ON)
option(ENABLE_EXAMPLE "Enable example" ON)
option(ENABLE_BENCHMARK "Enable benchmark" ON)
option(ENABLE_TESTS "Enable testing" ON)
option(ENABLE_CUDA_LAUNCH "Enable cuda kernel launch" OFF)
option(USE_PLATFORM_WAYLAND "Enable Wayland" OFF)
option(USE_PLATFORM_XCB "Enable XCB" OFF)
option(USE_PLATFORM_XLIB "Enable XLIB" OFF)
option(USE_PLATFORM_WIN32 "Enable WIN32" OFF)
option(USE_PLATFORM_ANDROID "Enable ANDROID" OFF)
option(USE_PLATFORM_METAL "Enable Metal" OFF)

if(USE_VALIDATION_LAYERS)
    target_compile_definitions(vkop PRIVATE USE_VALIDATION_LAYERS)
endif()
if(USE_DEBUG_LAYERS)
    target_compile_definitions(vkop PRIVATE USE_DEBUG_LAYERS)
endif()
if(USE_MEASURE_TIME)
    target_compile_definitions(vkop PUBLIC USE_MEASURE_TIME)
endif()
if(USE_VMA)
    target_compile_definitions(vkop PUBLIC USE_VMA)
endif()
if(USE_FP16)
    target_compile_definitions(vkop PUBLIC FP16)
endif()
if (ENABLE_CUDA_LAUNCH)
    target_compile_definitions(vkop PUBLIC ENABLE_CUDA_LAUNCH)
endif()

if(USE_PLATFORM_WAYLAND)
    target_compile_definitions(vkop PUBLIC VK_USE_PLATFORM_WAYLAND_KHR)
endif()
if(USE_PLATFORM_XCB)
    target_compile_definitions(vkop PUBLIC VK_USE_PLATFORM_XCB_KHR)
endif()
if(USE_PLATFORM_XLIB)
    target_compile_definitions(vkop PUBLIC VK_USE_PLATFORM_XLIB_KHR)
endif()
if(USE_PLATFORM_WIN32)
    target_compile_definitions(vkop PUBLIC VK_USE_PLATFORM_WIN32_KHR)
endif()
if(USE_PLATFORM_ANDROID)
    target_compile_definitions(vkop PUBLIC VK_USE_PLATFORM_ANDROID_KHR)
endif()
if(USE_PLATFORM_METAL)
    target_compile_definitions(vkop PUBLIC VK_USE_PLATFORM_METAL_EXT)
endif()


option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(ENABLE_ASAN)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
            message(FATAL_ERROR "GCC >= 4.8 required for AddressSanitizer")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.1)
            message(FATAL_ERROR "Clang >= 3.1 required for AddressSanitizer")
        endif()
    else()
        message(WARNING "AddressSanitizer not supported for this compiler")
    endif()

    target_compile_options(vkop PUBLIC
        -fsanitize=address
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(vkop PUBLIC
        -fsanitize=address
    )

    set(ENV{ASAN_OPTIONS} "detect_leaks=1")

    message(STATUS "AddressSanitizer enabled for target: your_target_name")
endif()

if (ENABLE_EXAMPLE)
add_subdirectory(example)
endif ()

if (ENABLE_BENCHMARK)
add_subdirectory(benchmark)
endif ()

if (ENABLE_TESTS)
enable_testing()
add_subdirectory(tests)
endif()
