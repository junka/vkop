cmake_minimum_required(VERSION 3.11)

project(vkop)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wextra -Werror -fno-omit-frame-pointer")

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/ops)

# Source files
file(GLOB_RECURSE SRC_FILES main.cpp src/*.cpp src/*.hpp ops/*.cpp ops/*.hpp)

# Add executable
add_executable(vkop ${SRC_FILES})

# Shader compilation
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(GLOB SHADER_FILES ${SHADER_DIR}/*.comp ${SHADER_DIR}/*.vert)

# Ensure SPIRV output directory exists
file(MAKE_DIRECTORY ${SPIRV_DIR})

# Compile shaders to SPIR-V and convert to header
foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPIRV_FILE ${SPIRV_DIR}/${SHADER_NAME}.spv)
    set(HEADER_FILE ${SPIRV_DIR}/${SHADER_NAME}.h)

    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND glslc -c ${SHADER} -o ${SPIRV_FILE}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )

    add_custom_command(
        OUTPUT ${HEADER_FILE}
        COMMAND xxd -i ${SPIRV_FILE} | sed "s/[_a-z0-9]*${SHADER_NAME}/${SHADER_NAME}/g" > ${HEADER_FILE}
        DEPENDS ${SPIRV_FILE}
        COMMENT "Converting ${SPIRV_FILE} to header file"
    )

    list(APPEND SPIRV_HEADERS ${HEADER_FILE})
endforeach()

# Add generated headers to the target
add_custom_target(compile_shaders DEPENDS ${SPIRV_HEADERS})
add_dependencies(vkop compile_shaders)

# Include SPIR-V headers
target_include_directories(vkop PRIVATE ${SPIRV_DIR})
target_link_libraries(vkop PRIVATE ${CMAKE_DL_LIBS})