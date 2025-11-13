cmake_minimum_required(VERSION 3.11)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wextra -Werror -fno-omit-frame-pointer")

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

# Source files
file(GLOB_RECURSE SRC_FILES vulkan/*.cpp ops/*.cpp core/*.cpp)
file(GLOB_RECURSE ALL_SOURCE_FILES vulkan/*.cpp vulkan/*.hpp ops/*.cpp core/*.cpp ops/*.hpp core/*.hpp)

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
    set(SPIRV_SOURCE_FILE ${SPIRV_DIR}/${SHADER_NAME}.c)

    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND glslc -c ${SHADER} -fshader-stage=compute -g -o ${SPIRV_FILE}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )

    add_custom_command(
        OUTPUT ${SPIRV_SOURCE_FILE}
        COMMAND xxd -i ${SPIRV_FILE} | sed "s/[_a-zA-Z0-9]*${SHADER_NAME}/${SHADER_NAME}/g" > ${SPIRV_SOURCE_FILE}
        DEPENDS ${SPIRV_FILE}
        COMMENT "Converting ${SPIRV_FILE} to header file"
    )

    list(APPEND SPIRV_HEADERS ${SPIRV_SOURCE_FILE})
    list(APPEND SRC_FILES ${SPIRV_SOURCE_FILE})
endforeach()


# Add executable
add_library(vkop ${SRC_FILES})

# Ensure whole-archive is used when linking vkop

# Add generated headers to the target
add_custom_target(compile_shaders DEPENDS ${SPIRV_HEADERS})
add_dependencies(vkop compile_shaders)

# Include SPIR-V headers
target_include_directories(vkop PRIVATE ${SPIRV_DIR})
target_link_libraries(vkop PRIVATE ${CMAKE_DL_LIBS})

add_subdirectory(model)

add_executable(vkopmain main.cpp)
target_link_libraries(vkopmain -Wl,--whole-archive vkop -Wl,--no-whole-archive vload)


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
option(USE_MEASURE_TIME "Enable measuring time" OFF)
option(USE_VMA "Enable usage of Vulkan Memory Allocator" OFF)

if(USE_VMA)
    target_compile_definitions(vkop PUBLIC USE_VMA)
endif()

enable_testing()

add_subdirectory(tests)