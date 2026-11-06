add_executable(vkyolo main.cpp)
target_link_libraries(vkyolo PRIVATE vkop vload)
target_compile_features(vkyolo PRIVATE cxx_std_17)

find_package(Vulkan REQUIRED)
target_link_libraries(vkyolo PRIVATE Vulkan::Vulkan)

# Dev archive fallback: vkopbin files embed SPIR-V subsets, but a dev binary
# that runs arbitrary models during iteration needs the full build tree archive.
# libvkop carries no SPIR-V of its own; every shader byte comes from either
# the vkopbin's embedded archive or this fallback. See include/dev_shaders.hpp.
target_compile_definitions(vkyolo PRIVATE
    "VKOP_DEV_SHADER_ARCHIVE=\"${VKOP_SHADER_ARCHIVE_FILE}\"")
add_dependencies(vkyolo shader_archive)

# Shader compilation for the fullscreen-blit renderer. Same pattern as
# example/demo/video: compile quad.vert + image.frag into SPIR-V, embed as
# C arrays with xxd -i. Shader sources live in example/demo/video/shaders/
# (we reuse them rather than duplicate), so we need to point glslc there.
find_program(GLSLC_EXECUTABLE glslc)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found — needed for yolo renderer shader")
endif()

set(VIDEO_SHADER_DIR ${CMAKE_SOURCE_DIR}/example/demo/video/shaders)
set(VIDEO_SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${VIDEO_SPIRV_DIR})

set(YOLO_SHADER_SRCS
    ${VIDEO_SHADER_DIR}/quad.vert
    ${VIDEO_SHADER_DIR}/image.frag
)

foreach(SHADER ${YOLO_SHADER_SRCS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER} EXT)
    if(SHADER_EXT STREQUAL ".vert")
        set(SHADER_STAGE "-fshader-stage=vertex")
    elseif(SHADER_EXT STREQUAL ".frag")
        set(SHADER_STAGE "-fshader-stage=fragment")
    endif()

    set(SPV_FILE ${VIDEO_SPIRV_DIR}/${SHADER_NAME}.spv)
    set(SPV_SRC  ${VIDEO_SPIRV_DIR}/${SHADER_NAME}.c)

    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND ${GLSLC_EXECUTABLE} ${SHADER_STAGE} -g --target-spv=spv1.3 -o ${SPV_FILE} ${SHADER}
        DEPENDS ${SHADER}
        COMMENT "Compiling yolo ${SHADER_NAME} shader"
    )

    add_custom_command(
        OUTPUT ${SPV_SRC}
        COMMAND xxd -i ${SPV_FILE} | sed "s/[_a-zA-Z0-9]*${SHADER_NAME}/${SHADER_NAME}/g" > ${SPV_SRC}
        DEPENDS ${SPV_FILE}
        COMMENT "Embedding yolo ${SHADER_NAME}.spv into C array"
    )

    set_source_files_properties(${SPV_SRC} PROPERTIES LANGUAGE C)
    list(APPEND YOLO_SPIRV_FILES ${SPV_SRC})
endforeach()

target_sources(vkyolo PRIVATE ${YOLO_SPIRV_FILES})

# GLFW: same as example/demo/video — prefer system install.
option(YOLO_USE_GLFW "Use GLFW for window in yolo example" ON)
if(YOLO_USE_GLFW)
    find_package(glfw3 QUIET)
    if(glfw3_FOUND)
        target_link_libraries(vkyolo PRIVATE glfw)
        target_compile_definitions(vkyolo PRIVATE USE_GLFW)
        message(STATUS "[yolo] GLFW found — window path enabled")
    else()
        message(STATUS "[yolo] GLFW not found — window path disabled, offscreen still works")
    endif()
endif()

# FreeType for real glyph rendering of COCO labels.
# -DYOLO_USE_FREETYPE=ON (default)  → requires find_package(Freetype);
#   if libfreetype-dev is not installed, build fails with a clear message
#   telling the user either to install it or to pass -DYOLO_USE_FREETYPE=OFF.
# -DYOLO_USE_FREETYPE=OFF → FreeType is not linked and label drawing is
#   compiled out; the binary still builds (you get a window without tags).
option(YOLO_USE_FREETYPE "Render COCO labels with FreeType" ON)
if(YOLO_USE_FREETYPE)
    find_package(Freetype REQUIRED
        "libfreetype not found — install libfreetype-dev (apt), or pass -DYOLO_USE_FREETYPE=OFF
         to build vkyolo without label rendering (no tags drawn on boxes).")
    target_link_libraries(vkyolo PRIVATE Freetype::Freetype)
    message(STATUS "[yolo] FreeType found — real glyph labels enabled")
else()
    message(STATUS "[yolo] YOLO_USE_FREETYPE=OFF — label rendering compiled out")
    target_compile_definitions(vkyolo PRIVATE YOLO_NO_LABELS)
endif()

# V4L2 needs libv4l2 optionally; plain ioctls work without it.
target_link_libraries(vkyolo PRIVATE dl)   # for dlsym fallback if needed
target_link_libraries(vkyolo PRIVATE pthread)
