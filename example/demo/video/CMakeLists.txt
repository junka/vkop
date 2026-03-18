option(USE_GLFW "Use GLFW for window creation" ON)
option(GLFW_USE_SYSTEM "Use system installed GLFW" ON)

find_package(Vulkan REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wall -Wextra -Werror -fno-omit-frame-pointer ${FP16_CXX_FLAGS}")

if(USE_GLFW)
    if(GLFW_USE_SYSTEM)
        find_package(glfw3 REQUIRED)
        message(STATUS "Using system GLFW")
    else()
        include(FetchContent)
        
        FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.3.8
            GIT_SHALLOW TRUE
        )
        
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

        if(ANDROID)
            set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
            set(GLFW_BUILD_X11 OFF CACHE BOOL "" FORCE)
        endif()
        
        FetchContent_MakeAvailable(glfw)
        message(STATUS "Using downloaded GLFW")
    endif()
endif()


# Shader compilation for video shaders
set(VIDEO_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(VIDEO_SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${VIDEO_SPIRV_DIR})

file(GLOB VIDEO_SHADER_FILES 
    ${VIDEO_SHADER_DIR}/*.vert 
    ${VIDEO_SHADER_DIR}/*.frag
)

foreach(SHADER ${VIDEO_SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER} EXT)
    
    # Determine shader stage
    if(SHADER_EXT STREQUAL ".vert")
        set(SHADER_STAGE "vertex")
        set(GLSLC_STAGE "-fshader-stage=vertex")
    elseif(SHADER_EXT STREQUAL ".frag")
        set(SHADER_STAGE "fragment")
        set(GLSLC_STAGE "-fshader-stage=fragment")
    endif()
    
    set(SPIRV_FILE ${VIDEO_SPIRV_DIR}/${SHADER_NAME}.spv)
    set(SPIRV_SOURCE_FILE ${VIDEO_SPIRV_DIR}/${SHADER_NAME}.c)
    
    # Compile to SPIR-V
    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND ${GLSLC_EXECUTABLE} ${GLSLC_STAGE} -g --target-spv=spv1.3 -o ${SPIRV_FILE} ${SHADER}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME}.${SHADER_STAGE} to SPIR-V"
    )
    
    # Convert SPIR-V to C header using the same script as parent project
    if(WIN32)
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
    list(APPEND VIDEO_SPIRV_HEADERS ${SPIRV_SOURCE_FILE})
    list(APPEND VIDEO_SRC_FILES ${SPIRV_SOURCE_FILE})
endforeach()

add_executable(vkdeo main.cpp 
    ${VIDEO_SRC_FILES})
target_link_libraries(vkdeo PRIVATE vkop)

if(USE_GLFW)
    target_link_libraries(vkdeo PUBLIC glfw)
    target_compile_definitions(vkdeo PUBLIC USE_GLFW)
endif()

if(ANDROID)
    target_link_libraries(vkdeo PRIVATE android log)
endif()