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
    set(FP16_CXX_FLAGS "-mfp16-format=ieee")
    # set(FP16_CXX_FLAGS "-march=armv8.2-a+fp16")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM")
    set(FP16_CXX_FLAGS "-mfp16-format=ieee")
else()
    message(WARNING "FP16 not supported on this architecture")
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
            set(Vulkan_LIBRARIES ${VULKAN_SDK_PATH}/lib/libvulkan.so)
            set(Vulkan_FOUND TRUE)

            link_directories(${VULKAN_SDK_PATH}/lib)
        else()
            message(FATAL_ERROR "Cannot find Vulkan headers or libraries in ${VULKAN_SDK_PATH}")
        endif()
    endif()
else()
    # Fall back to system default paths
    find_package(Vulkan REQUIRED)
endif()
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

# Source files
file(GLOB_RECURSE SRC_FILES vulkan/*.cpp ops/*.cpp core/*.cpp)
file(GLOB_RECURSE ALL_SOURCE_FILES vulkan/*.cpp vulkan/*.hpp ops/*.cpp core/*.cpp ops/*.hpp core/*.hpp)

# Shader compilation
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(GLOB SHADER_FILES ${SHADER_DIR}/*.comp ${SHADER_DIR}/*.vert)
# Exclude common files from SHADER_FILES
list(FILTER SHADER_FILES EXCLUDE REGEX "activation\\.comp$")

# Ensure SPIRV output directory exists
file(MAKE_DIRECTORY ${SPIRV_DIR})

option(USE_FP16 "Enable usage of FP16" OFF)
if(USE_FP16)
    set(SHADER_FLAGS "-DFP16")
endif()
# Compile shaders to SPIR-V and convert to header
foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPIRV_FILE ${SPIRV_DIR}/${SHADER_NAME}.spv)
    set(SPIRV_SOURCE_FILE ${SPIRV_DIR}/${SHADER_NAME}.c)

    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND glslc -c ${SHADER} -fshader-stage=compute -g --target-spv=spv1.3 -o ${SPIRV_FILE} ${SHADER_FLAGS}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )
if(WIN32)
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
add_custom_command(
    OUTPUT ${SPIRV_SOURCE_FILE}
    COMMAND ${Python3_EXECUTABLE} 
        ${PY_SCRIPT_FILE}
        ${SPIRV_FILE}
        ${SPIRV_SOURCE_FILE}
        ${SHADER_NAME}
    DEPENDS 
        ${SPIRV_FILE}
        ${PY_SCRIPT_FILE}
    COMMENT "Embedding SPIR-V shader ${SHADER_NAME} into C header"
)
else()
    add_custom_command(
        OUTPUT ${SPIRV_SOURCE_FILE}
        COMMAND xxd -i ${SPIRV_FILE} | sed "s/[_a-zA-Z0-9]*${SHADER_NAME}/${SHADER_NAME}/g" > ${SPIRV_SOURCE_FILE}
        DEPENDS ${SPIRV_FILE}
        COMMENT "Converting ${SPIRV_FILE} to header file"
    )
endif()
    set_source_files_properties(${SPIRV_SOURCE_FILE} PROPERTIES LANGUAGE C)
    list(APPEND SPIRV_HEADERS ${SPIRV_SOURCE_FILE})
    list(APPEND SRC_FILES ${SPIRV_SOURCE_FILE})
endforeach()


# Add executable
add_library(vkop ${SRC_FILES})
message(${SRC_FILES})

# Ensure whole-archive is used when linking vkop

# Add generated headers to the target
add_custom_target(compile_shaders DEPENDS ${SPIRV_HEADERS})
add_dependencies(vkop compile_shaders)

# Include SPIR-V headers
target_include_directories(vkop PRIVATE ${SPIRV_DIR})
target_link_libraries(vkop PRIVATE ${CMAKE_DL_LIBS})

add_subdirectory(model)

add_executable(vkopmain main.cpp)
target_link_libraries(vkopmain vkop vload -lpthread)


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

  add_dependencies(vkopmain format-code)
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
    # add_dependencies(vkopmain lint-code)
endif()

option(USE_VALIDATION_LAYERS "Enable validation layers" OFF)
option(USE_DEBUG_LAYERS "Enable debug layers" OFF)
option(USE_MEASURE_TIME "Enable measuring time" OFF)
option(USE_VMA "Enable usage of Vulkan Memory Allocator" ON)
option(ENABLE_EXAMPLE "Enable example" ON)
option(ENABLE_BENCHMARK "Enable benchmark" ON)
option(ENABLE_TESTS "Enable testing" ON)

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
