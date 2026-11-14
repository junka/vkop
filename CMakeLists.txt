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
# /utf-8: sources are UTF-8 without a BOM; decoded as the GBK system codepage
# a trailing multibyte comment byte can swallow the newline and shift the
# parser, which surfaces as bogus "missing type specifier" errors.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od /Zi /W3 /fp:fast /utf-8 /bigobj")
# /FS: /MP spawns several cl.exe per project that all write the same .pdb;
# without it every incremental rebuild loses the race (error C1041).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FS")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wall -Wextra -Werror -fno-omit-frame-pointer ${FP16_CXX_FLAGS}")
if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Warnings clang adds over GCC that the codebase (and vendored VMA /
    # Vulkan headers) trip everywhere — -Werror from GCC's perspective is
    # still in effect for real issues:
    #  - nullability-completeness: Vulkan headers carry _Nullable/_Nonnull so
    #    clang infers it for every un-annotated pointer
    #  - unused-private-field / unused-lambda-capture: clang-only, GCC has no
    #    equivalent, so these fire on the brew-clang build only
    string(APPEND CMAKE_CXX_FLAGS " -Wno-nullability-completeness")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-private-field")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-lambda-capture")
    # clang 21+: fires on ~shared_ptr<ITensor> destroying Tensor<T>. ITensor's
    # dtor IS virtual, so polymorphic deletion is well-defined here.
    string(APPEND CMAKE_CXX_FLAGS " -Wno-delete-non-abstract-non-virtual-dtor")
endif()
endif()

# VK_EXT_shader_float8 support: the fp8 weight kernels can decode bytes with
# the hardware conversion instead of reassembling the fields by hand, but only
# on a device that exposes the extension AND with a shader compiler that knows
# GL_EXT_float_e4m3. ON builds the modules whenever this glslc can; the runtime
# gate still decides per device. See docs/vulkan-1.4-side-by-side.md.
option(VKOP_ENABLE_FLOAT8 "Build the VK_EXT_shader_float8 fp8 decode modules" ON)

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
# The SwiGLU fold is built by its own rule below (a named key, no folder
# prefix); leaving it in the generic loop would register a second, unused
# module under the same key.
list(FILTER SHADER_FILES EXCLUDE REGEX "/matmul_down_act\\.comp$")
# matmul_coop16_inwg.comp (a b3 SPLITK_INWG experiment) is dead: the host
# route is disabled (coopmat_inwg_applies returns false). Keep it out of the
# generic loop -- the explicit vkop_shader_variant entry below builds it with
# -DSPLITK_INWG, which the loop would otherwise compile without the flag.
list(FILTER SHADER_FILES EXCLUDE REGEX "/matmul_coop16_inwg\\.comp$")
# matmul_coop16_sklast.comp carries the SK_LASTBLOCK (+ optional ENABLE_HALF2_EPI)
# implementation as a complete separate source -- it has its own STORE_C macro,
# scatter_tile helpers, and a three-way main() split. Generic loop exclusion
# lets sklast variants compile with -DSK_LASTBLOCK (and optionally -DENABLE_HALF2_EPI).
list(FILTER SHADER_FILES EXCLUDE REGEX "/matmul_coop16_sklast\\.comp$")
# argmax.comp is a kept reference kernel backing no engine op (GPU argmax
# rides TopK k=1, see the llm_chat target below), so the generic loop must
# not claim it into vkop_shaders.bin and nothing compiles it today. The FP16
# macro selects packed-fp16 vs fp32 logits reads.
list(FILTER SHADER_FILES EXCLUDE REGEX "/argmax\\.comp$")

# glslc resolves #include but cmake does not track it, so editing a shared
# header would silently rebuild nothing. Track them explicitly.
file(GLOB SHADER_INCLUDES ${SHADER_DIR}/common/*.comp)

# Ensure SPIRV output directory exists
file(MAKE_DIRECTORY ${SPIRV_DIR})

# No USE_FP16 knob: fp16 kernels are per-shader dual builds (DUAL_FP16_SHADERS)
# selected from the tensor dtype in the operators, so a build flag cannot make
# a graph fp16 or fp32 any more. The caller picks the graph precision (the
# Runtime ctor's precision argument, defaulting to fp32).

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Compile shaders to SPIR-V. Nothing about a shader is linked into libvkop:
# libvkop carries no SPIR-V of its own, and each op resolves its shader by key
# from the archive the model's .vkopbin carries (Operator::resolve_primary_spv).
# Only the .spv files matter here, as inputs to the archive below. The key is
# folder-derived: shaders/image/relu.comp -> image_relu,
# shaders/buffer/unary.comp -> buffer_unary, so two same-named shaders in
# different folders never collide, and the same name ids the model's embedded
# subset (onnx2vkop copies from this archive at convert time).
#
# Development binaries that build ops without a model take this build tree's
# archive via include/dev_shaders.hpp; there is no env override, so a shipping
# run always executes the kernels its own file names.

foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    file(RELATIVE_PATH SHADER_REL ${SHADER_DIR} ${SHADER})
    string(REPLACE "/" "_" SYM_BASE "${SHADER_REL}")
    get_filename_component(SYM_BASE "${SYM_BASE}" NAME_WE)
    # SPIRV outputs are folder-prefixed so two same-named shaders in different
    # folders (image/concat vs buffer/concat) don't collide.
    set(SPIRV_FILE ${SPIRV_DIR}/${SYM_BASE}.spv)

    # Both folders ship fp16 as a separate dual build (DUAL_FP16_SHADERS
    # below), so the base build here is always the fp32 variant — never pass
    # -DFP16 (it would make the base build the fp16 variant, which for the
    # image path silently re-types every accumulator to float16_t regardless of
    # the tensor dtype, since the image ops bind one module for all dtypes).
    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER} -fshader-stage=compute -g --target-spv=spv1.3 -o ${SPIRV_FILE}
        DEPENDS ${SHADER} ${SHADER_INCLUDES}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE})
endforeach()

# Some shaders keep separate fp32/fp16 SPIR-V builds (no runtime branch in
# the shader; the operator picks the right one at construction). Each entry is
# a path relative to SHADER_DIR (e.g. buffer/gather, buffer/unary). The fp16
# build is compiled with -DFP16 and archived as <folder>_<name>_fp16
# (folder-derived, same as the base build).
#
# fp16 buffer storage packs two half values per uint word, so the fp16 path
# must use a word-per-thread store pattern to avoid read-modify-write races.
# Transpose is non-contiguous (permuted) so its fp16 path reads each output
# element via the packed-half BUF_LOAD and packs two results per output word.
set(DUAL_FP16_SHADERS
    # The three qwen3_5 linear-attention ops. Their _fp16 build uses
    # float16_t storage; the extra _fp16_packed variant (built below) is the
    # two-halves-per-word convention the rest of this repo uses, kept so the
    # two can be measured against each other.
    buffer/cumsum
    buffer/conv1d
    buffer/linear_attention
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
    buffer/globalaveragepool
    buffer/prelu
    buffer/reshape
    buffer/expand
    buffer/resize
    buffer/transpose
    buffer/concat
    buffer/split
    buffer/softmax
    buffer/matmul
    buffer/cos
    buffer/sin
    buffer/neg
    buffer/erf
    buffer/relu
    buffer/floor
    buffer/conv2d
    buffer/rotary
    buffer/fused_elemwise
    buffer/rmsnorm
    buffer/layernorm
    buffer/attention
    buffer/slice
    buffer/scatter_elements
    buffer/min
    buffer/max
    buffer/mod
    # maxpool2d.comp has the FP16 packing path and ShaderKeys/shader_map
    # already declare the buffer fp16 variant; without this entry an fp16
    # buffer MaxPool2d (VKOP_BUFFER_BACKEND=1) had no key to resolve.
    buffer/maxpool2d
    # Image kernels accumulate in fp32 (ACC types in the .comp), so the fp16
    # variant only changes the load/store width. The op picks the module from
    # the tensor dtype; a fp32 graph must never land on the fp16 module.
    image/add
    image/conv2d
    image/maxpool2d
    image/softmax
)
foreach(DUAL_NAME ${DUAL_FP16_SHADERS})
    string(REPLACE "/" "_" DUAL_BASE "${DUAL_NAME}")
    set(SPIRV_FILE_FP16 ${SPIRV_DIR}/${DUAL_BASE}_fp16.spv)
    add_custom_command(
        OUTPUT ${SPIRV_FILE_FP16}
        COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/${DUAL_NAME}.comp
                -fshader-stage=compute -g --target-spv=spv1.3
                -o ${SPIRV_FILE_FP16} -DFP16
        DEPENDS ${SHADER_DIR}/${DUAL_NAME}.comp ${SHADER_INCLUDES}
        COMMENT "Compiling ${DUAL_NAME} (fp16) to SPIR-V"
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_FP16})
endforeach()

# A second fp16 build of buffer/matmul.comp, entered through that file's
# -DW8_WIDE entry point (the int8/fp8 weight-only WIDE GEMV). Same source, same
# push-constant layout, so the two modules can never drift apart; SPIR-V DCE
# drops every other kernel from each build. It is a separate module because
# MoltenVK cannot compile that one kernel: as part of the general fp16 module its
# pipeline creation failed and took every buffer MatMul down with it (see the
# W8_WIDE entry in shaders/buffer/matmul.comp and MatMulBuffer::w8_wide_pipeline).
set(SPIRV_FILE_W8WIDE ${SPIRV_DIR}/buffer_matmul_w8_wide_fp16.spv)
add_custom_command(
    OUTPUT ${SPIRV_FILE_W8WIDE}
    COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/matmul.comp
            -fshader-stage=compute -g --target-spv=spv1.3
            -o ${SPIRV_FILE_W8WIDE} -DFP16 -DW8_WIDE
    DEPENDS ${SHADER_DIR}/buffer/matmul.comp ${SHADER_INCLUDES}
    COMMENT "Compiling buffer/matmul (int8 weight-only WIDE, fp16) to SPIR-V"
)
list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_W8WIDE})

# The fp8 hardware-decode module (-DFP8_HW on buffer/matmul.comp). Compiled
# ONLY when this glslc can emit GL_EXT_float_e4m3, because the shader requires
# that extension and an older shaderc rejects the source outright. This is the
# shader-compiler half of the VK_EXT_shader_float8 story; the device half is a
# separate runtime gate in VulkanDevice::createFeatureDescriptors, and the two
# must agree -- the host builds the pipeline only when both are present.
#
# The probe writes an empty .comp that requires the extension and checks
# whether a .spv came out: glslc exits 0 even on a compile error (measured on
# shaderc 2023.8), so the exit code cannot be the signal.
set(VKOP_GLSLC_HAS_FLOAT8 FALSE)
if(VKOP_ENABLE_FLOAT8)
    set(_f8_probe_src ${CMAKE_CURRENT_BINARY_DIR}/glslc_float8_probe.comp)
    set(_f8_probe_out ${CMAKE_CURRENT_BINARY_DIR}/glslc_float8_probe.spv)
    file(WRITE ${_f8_probe_src}
        "#version 460\n"
        "#extension GL_EXT_float_e4m3 : require\n"
        "layout(local_size_x = 32) in;\n"
        "void main() {}\n")
    file(REMOVE ${_f8_probe_out})
    execute_process(
        COMMAND ${GLSLC_EXECUTABLE} -c ${_f8_probe_src}
                -fshader-stage=compute -o ${_f8_probe_out}
        OUTPUT_QUIET ERROR_QUIET)
    if(EXISTS ${_f8_probe_out})
        set(VKOP_GLSLC_HAS_FLOAT8 TRUE)
        message(STATUS "glslc accepts GL_EXT_float_e4m3: building the fp8 hardware-decode module")
    else()
        message(STATUS "glslc rejects GL_EXT_float_e4m3 (shaderc too old): "
                       "the fp8 hardware-decode module is NOT built; fp8 weights "
                       "keep the software decode. Use a Vulkan SDK >= 1.4.321.")
    endif()
endif()

if(VKOP_GLSLC_HAS_FLOAT8)
    # Two builds, one per entry point the host can select: the general fp16
    # module and the WIDE GEMV one. Same -DW8_WIDE split as the software
    # modules above, so the hardware decode reaches the path that actually runs
    # at decode time (ksplit_w8wide) and not just the fallback.
    set(SPIRV_FILE_F8 ${SPIRV_DIR}/buffer_matmul_fp8_hw_fp16.spv)
    add_custom_command(
        OUTPUT ${SPIRV_FILE_F8}
        COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/matmul.comp
                -fshader-stage=compute -g --target-spv=spv1.3
                -o ${SPIRV_FILE_F8} -DFP16 -DFP8_HW
        DEPENDS ${SHADER_DIR}/buffer/matmul.comp ${SHADER_INCLUDES}
        COMMENT "Compiling buffer/matmul (fp8 hardware decode, fp16) to SPIR-V"
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_F8})

    set(SPIRV_FILE_F8_WIDE ${SPIRV_DIR}/buffer_matmul_w8_wide_fp8_hw_fp16.spv)
    add_custom_command(
        OUTPUT ${SPIRV_FILE_F8_WIDE}
        COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/matmul.comp
                -fshader-stage=compute -g --target-spv=spv1.3
                -o ${SPIRV_FILE_F8_WIDE} -DFP16 -DW8_WIDE -DFP8_HW
        DEPENDS ${SHADER_DIR}/buffer/matmul.comp ${SHADER_INCLUDES}
        COMMENT "Compiling buffer/matmul (fp8 hardware decode, WIDE, fp16) to SPIR-V"
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_F8_WIDE})

    # The Conv2d fp8 hardware-decode module, in the two builds the op selects
    # between (fp32 and fp16 output). Same -DFP8_HW treatment as matmul: a
    # separate module because OpCapability Float8EXT is module-wide.
    set(SPIRV_FILE_CONV_F8 ${SPIRV_DIR}/buffer_conv2d_fp8_hw.spv)
    add_custom_command(
        OUTPUT ${SPIRV_FILE_CONV_F8}
        COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/conv2d.comp
                -fshader-stage=compute -g --target-spv=spv1.3
                -o ${SPIRV_FILE_CONV_F8} -DFP8_HW
        DEPENDS ${SHADER_DIR}/buffer/conv2d.comp ${SHADER_INCLUDES}
        COMMENT "Compiling buffer/conv2d (fp8 hardware decode) to SPIR-V"
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_CONV_F8})

    set(SPIRV_FILE_CONV_F8_FP16 ${SPIRV_DIR}/buffer_conv2d_fp8_hw_fp16.spv)
    add_custom_command(
        OUTPUT ${SPIRV_FILE_CONV_F8_FP16}
        COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/conv2d.comp
                -fshader-stage=compute -g --target-spv=spv1.3
                -o ${SPIRV_FILE_CONV_F8_FP16} -DFP16 -DFP8_HW
        DEPENDS ${SHADER_DIR}/buffer/conv2d.comp ${SHADER_INCLUDES}
        COMMENT "Compiling buffer/conv2d (fp8 hardware decode, fp16) to SPIR-V"
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_CONV_F8_FP16})
endif()

# Two 512-lane split-K builds of buffer/matmul.comp (VKOP_KSPLIT512=1 host
# switch): the ksplit-only general module and the int8 WIDE module. Same source
# and push constants as the 256-lane builds; only local_size (32x16), the
# KS_LANES slice depth and the entry differ. Host probes each key and falls
# back to the 256-lane pipeline where the driver or the model archive refuses.
set(SPIRV_FILE_KS512 ${SPIRV_DIR}/buffer_matmul_ksplit512_fp16.spv)
add_custom_command(
    OUTPUT ${SPIRV_FILE_KS512}
    COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/matmul.comp
            -fshader-stage=compute -g --target-spv=spv1.3
            -o ${SPIRV_FILE_KS512} -DFP16 -DKSPLIT512
    DEPENDS ${SHADER_DIR}/buffer/matmul.comp ${SHADER_INCLUDES}
    COMMENT "Compiling buffer/matmul (512-lane ksplit, fp16) to SPIR-V"
)
list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_KS512})
set(SPIRV_FILE_KS512_W8WIDE ${SPIRV_DIR}/buffer_matmul_ksplit512_w8_wide_fp16.spv)
add_custom_command(
    OUTPUT ${SPIRV_FILE_KS512_W8WIDE}
    COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/matmul.comp
            -fshader-stage=compute -g --target-spv=spv1.3
            -o ${SPIRV_FILE_KS512_W8WIDE} -DFP16 -DW8_WIDE -DKSPLIT512
    DEPENDS ${SHADER_DIR}/buffer/matmul.comp ${SHADER_INCLUDES}
    COMMENT "Compiling buffer/matmul (512-lane int8 WIDE, fp16) to SPIR-V"
)
list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_KS512_W8WIDE})

# The tile-split-K pair for matmul16_tiled: one source, two modules. A K-range
# carried across workgroups needs a fold pass, and a pipeline binds exactly one
# entry point, so the same .comp is built twice - the deposit pass
# (-DTILED_KS_SPLIT) and the fold pass (-DTILED_KS_FOLD). The host dispatches
# split then fold with a storage barrier between them (MatMulBuffer::
# ks_split_pipeline / ks_fold_pipeline); either module missing leaves the plain
# per-tile kernel in charge.
set(SPIRV_FILE_KS_SPLIT ${SPIRV_DIR}/buffer_matmul_tiled_ks_split_fp16.spv)
add_custom_command(
    OUTPUT ${SPIRV_FILE_KS_SPLIT}
    COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/matmul.comp
            -fshader-stage=compute -g --target-spv=spv1.3
            -o ${SPIRV_FILE_KS_SPLIT} -DFP16 -DTILED_KS_SPLIT
    DEPENDS ${SHADER_DIR}/buffer/matmul.comp ${SHADER_INCLUDES}
    COMMENT "Compiling buffer/matmul (tiled tile-split-K deposit, fp16) to SPIR-V"
)
list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_KS_SPLIT})
set(SPIRV_FILE_KS_FOLD ${SPIRV_DIR}/buffer_matmul_tiled_ks_fold_fp16.spv)
add_custom_command(
    OUTPUT ${SPIRV_FILE_KS_FOLD}
    COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/matmul.comp
            -fshader-stage=compute -g --target-spv=spv1.3
            -o ${SPIRV_FILE_KS_FOLD} -DFP16 -DTILED_KS_FOLD
    DEPENDS ${SHADER_DIR}/buffer/matmul.comp ${SHADER_INCLUDES}
    COMMENT "Compiling buffer/matmul (tiled tile-split-K fold, fp16) to SPIR-V"
)
list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_KS_FOLD})

# The SwiGLU down-projection fold (-DDOWN_ACT): one MatMul that absorbs the
# gate and up projections and their elementwise glue, so the M*N_up activation
# never reaches memory. A module of its own for the same reason W8_WIDE is --
# the fold needs two descriptors no other build declares, and a separate module
# keeps the shared fp16 pipeline untouched if a driver refuses it. Missing key
# => MatMulBuffer::down_act_pipeline yields nothing and the op falls back to the
# three standalone GEMMs.
# Two keys from one source. DA_M1_ONLY trims the multi-row tile walk, which
# every M==1 dispatch skips anyway; carrying it in the module the decode fold
# runs measured 4.2 ms/step slower (2026-11-05), so the decode key is built
# without it and the host routes M>1 to the _multi key.
set(SPIRV_FILE_DOWN_ACT ${SPIRV_DIR}/buffer_matmul_down_act_fp16.spv)
add_custom_command(
    OUTPUT ${SPIRV_FILE_DOWN_ACT}
    COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/buffer/matmul_down_act.comp
            -fshader-stage=compute -g --target-spv=spv1.3 -DDA_M1_ONLY
            -o ${SPIRV_FILE_DOWN_ACT}
    DEPENDS ${SHADER_DIR}/buffer/matmul_down_act.comp ${SHADER_INCLUDES}
    COMMENT "Compiling buffer/matmul_down_act (SwiGLU fold, M==1) to SPIR-V"
)
list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_DOWN_ACT})

# Extra SPIR-V builds of one shader source under explicit archive keys. The
# host resolves these by the same key it sees in the executable (MatMul's
# coopmat shapes and geometries, see MatMulBuffer::build_coop_pipeline and
# coop_geo_pipeline); the keys cannot be folder-derived because several modules
# come from the same .comp. A macro, not a function, so the SPIRV_OUTPUTS and
# VARIANT_* appends land in this directory's scope and reach the archive below.
macro(vkop_shader_variant KEY SOURCE)
    set(SPIRV_FILE_VARIANT ${SPIRV_DIR}/${KEY}.spv)
    add_custom_command(
        OUTPUT ${SPIRV_FILE_VARIANT}
        COMMAND ${GLSLC_EXECUTABLE} -c ${SHADER_DIR}/${SOURCE}.comp
                -fshader-stage=compute -g --target-spv=spv1.3
                -o ${SPIRV_FILE_VARIANT} ${ARGN}
        DEPENDS ${SHADER_DIR}/${SOURCE}.comp ${SHADER_INCLUDES}
        COMMENT "Compiling ${SOURCE} (${KEY}) to SPIR-V"
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE_VARIANT})
    list(APPEND VARIANT_SPV_KEYS ${KEY})
    list(APPEND VARIANT_SPV_FILES ${SPIRV_FILE_VARIANT})
endmacro()

# The three qwen3_5 linear-attention ops ship TWO fp16 layouts so their cost can
# be compared on a real device:
#   <base>_fp16         float16_t storage (native half loads/stores)
#   <base>_fp16_packed  two halves per uint word, unpack/pack per access
# The dual-fp16 loop above already produces the native one; the packed build is
# an extra variant, reachable through VKOP_FP16_PACKED=<op,..> at run time (see
# ops/Operator.cpp's shader-key override). Same source, same bytes on disk --
# only the indexing and the thread-per-word discipline differ.
foreach(PACKED_NAME buffer/cumsum buffer/conv1d buffer/linear_attention)
    string(REPLACE "/" "_" PACKED_BASE "${PACKED_NAME}")
    vkop_shader_variant(${PACKED_BASE}_fp16_packed ${PACKED_NAME} -DFP16 -DPACKED_FP16)
    vkop_shader_variant(${PACKED_BASE}_bda_fp16_packed ${PACKED_NAME} -DFP16 -DBDA -DPACKED_FP16)
endforeach()

# Device-address builds of the shared sources listed here. Each is compiled
# with -DBDA (and -DFP16 -DBDA when it also has a dual fp16 base), producing
# <base>_bda / <base>_bda_fp16 -- exactly what
# Operator::try_build_bda_pipeline probes. The operator always keeps the
# descriptor pipeline as the per-dispatch fallback, so a model converted
# without these keys simply stays on descriptors.
set(BDA_SHADERS
    buffer/cumsum
    buffer/conv1d
    buffer/linear_attention
    buffer/neg
    buffer/sigmoid
    buffer/tanh
    buffer/erf
    buffer/sqrt
    buffer/atan
    buffer/cos
    buffer/sin
    buffer/softplus
    buffer/relu
    buffer/floor
    buffer/rmsnorm
    buffer/layernorm
    buffer/cast
    buffer/quantize_linear
    buffer/softmax
    buffer/reduce
    buffer/transpose
    buffer/globalaveragepool
    buffer/rotary
    buffer/embedding_forward
    buffer/prelu
    buffer/resize
    buffer/expand
    buffer/add
    buffer/sub
    buffer/mul
    buffer/div
    buffer/pow
    buffer/min
    buffer/max
    buffer/mod
    buffer/matmul
    # Generic address-table ops: the -DBDA build declares ONE table at binding
    # 0 (see Operator::bda_address_table_slots) and keeps the descriptor PC.
    buffer/attention
    buffer/concat
    buffer/conv2d
    buffer/equal
    buffer/fused_elemwise
    buffer/gather
    buffer/gemm
    buffer/maxpool2d
    buffer/reshape
    buffer/scatter_elements
    buffer/scatternd
    buffer/slice
    buffer/split
    buffer/topk
    buffer/where
)
foreach(BDA_NAME ${BDA_SHADERS})
    string(REPLACE "/" "_" BDA_BASE "${BDA_NAME}")
    vkop_shader_variant(${BDA_BASE}_bda ${BDA_NAME} -DBDA)
    if(BDA_NAME IN_LIST DUAL_FP16_SHADERS)
        vkop_shader_variant(${BDA_BASE}_bda_fp16 ${BDA_NAME} -DFP16 -DBDA)
    endif()
endforeach()

# int64 shape-meta twins: separate keys the ops resolve by name, so they need
# their own -DBDA builds (no fp16 form). The host keeps the descriptor module
# as the fallback when the key is absent.
vkop_shader_variant(buffer_concat_int64_bda buffer/concat_int64 -DBDA)
vkop_shader_variant(buffer_equal_int64_bda buffer/equal_int64 -DBDA)
vkop_shader_variant(buffer_gather_int64_bda buffer/gather_int64 -DBDA)
vkop_shader_variant(buffer_where_int64_bda buffer/where_int64 -DBDA)
vkop_shader_variant(buffer_expand_int64_bda buffer/expand_int64 -DBDA)
# int64 -> fp32 Cast (position ids). Needs shaderInt64 for the ARITHMETIC, not
# the ivec2 view: this is the one int64 op that changes the element type.
vkop_shader_variant(buffer_cast_int64_bda buffer/cast_int64 -DBDA)
# int64 Transpose (the NonZero output). A pure ivec2 move, so unlike the Cast
# twin it does NOT need shaderInt64.
vkop_shader_variant(buffer_transpose_int64_bda buffer/transpose_int64 -DBDA)
# int64 elementwise binary (add/sub/mul/div). Unlike concat/where/gather/
# expand/transpose these are NOT byte moves: the result of two int64 operands
# is an int64, which needs real 64-bit arithmetic, so they require shaderInt64
# like the Cast twin. The host keeps its CPU path for devices without the
# feature (BufferBinaryFactory::cpuComputeInt64).
vkop_shader_variant(buffer_add_int64_bda buffer/add_int64 -DBDA)
vkop_shader_variant(buffer_sub_int64_bda buffer/sub_int64 -DBDA)
vkop_shader_variant(buffer_mul_int64_bda buffer/mul_int64 -DBDA)
vkop_shader_variant(buffer_div_int64_bda buffer/div_int64 -DBDA)
# int64 Range. A SEPARATE KEY from buffer_range on purpose: that module is the
# old host-era stub (four bound param buffers, 4-byte PC), and the archive a
# model carries was built against whichever version existed when it was
# converted. Host code that dispatches one and binds for the other produces
# all-zero output with no error — exactly the failure the old shader's header
# warns about. A distinct key means an old .vkopbin simply has no
# buffer_range_int64 and Range keeps its host generator, which is correct for
# both.
vkop_shader_variant(buffer_range_int64_bda buffer/range_int64 -DBDA)
# The GPU-shape indirect pre-pass (BufferBinaryFactory::submit_prepass), whose
# two bindings are the out-shape SSBO and the VkDispatchIndirectCommand buffer
# -- neither comes from the main kernel's objs_ list.
vkop_shader_variant(buffer_dispatch_from_shape_bda
                    buffer/dispatch_from_shape -DBDA)

# Host-only ops (NonZero/Range) and the reference argmax kernel back no engine
# dispatch, but their -DBDA builds are exercised directly by
# tests/DeadShaderBdaProbe.cpp so every shader in the tree has a BDA path.
vkop_shader_variant(buffer_nonzero_bda buffer/nonzero -DBDA)
vkop_shader_variant(buffer_range_bda buffer/range -DBDA)
vkop_shader_variant(buffer_argmax buffer/argmax)
vkop_shader_variant(buffer_argmax_bda buffer/argmax -DBDA)

# The M>1 build of the SwiGLU fold source (no DA_M1_ONLY): keeps the multi-row
# deposit walk that the M==1 module above compiles out.
vkop_shader_variant(buffer_matmul_down_act_multi_fp16 buffer/matmul_down_act)

# BDA address-table builds of the fold. The fold reads its nine data buffers
# through several element types (uint / uvec4 / uvec2), so its table carries
# one address per (buffer, view) at 16/8-byte-aligned slots; one 112 B table
# descriptor replaces all nine. Same -DDA_M1_ONLY split as the descriptor
# builds, and the host keeps both descriptor modules as the fallback.
vkop_shader_variant(buffer_matmul_down_act_bda_fp16 buffer/matmul_down_act
                    -DDA_M1_ONLY -DBDA)
vkop_shader_variant(buffer_matmul_down_act_multi_bda_fp16 buffer/matmul_down_act
                    -DBDA)

# BDA address-table builds of the same-source MatMul variants. Every one
# replaces the seven data descriptors with the single table descriptor; the
# host probes each key lazily and keeps the descriptor module as the fallback.
vkop_shader_variant(buffer_matmul_w8_wide_bda_fp16 buffer/matmul
                    -DFP16 -DW8_WIDE -DBDA)
vkop_shader_variant(buffer_matmul_ksplit512_bda_fp16 buffer/matmul
                    -DFP16 -DKSPLIT512 -DBDA)
vkop_shader_variant(buffer_matmul_ksplit512_w8_wide_bda_fp16 buffer/matmul
                    -DFP16 -DW8_WIDE -DKSPLIT512 -DBDA)
vkop_shader_variant(buffer_matmul_tiled_ks_split_bda_fp16 buffer/matmul
                    -DFP16 -DTILED_KS_SPLIT -DBDA)
vkop_shader_variant(buffer_matmul_tiled_ks_fold_bda_fp16 buffer/matmul
                    -DFP16 -DTILED_KS_FOLD -DBDA)
# The fp8 hardware-decode WIDE GEMV with BDA addressing. Without this key the
# BDA path keeps the software decode: the descriptor-buffer module is a
# different archive entry, so enabling BDA alone would silently drop the
# hardware conversion (measured: 113.8 -> 35.6 GB/s on Ada). Only defined when
# this glslc can compile the fp8 source at all.
if(VKOP_GLSLC_HAS_FLOAT8)
    vkop_shader_variant(buffer_matmul_w8_wide_bda_fp8_hw_fp16 buffer/matmul
                        -DFP16 -DW8_WIDE -DBDA -DFP8_HW)
    # 512-lane WIDE, with and without BDA. Same reasoning: these routes are
    # opt-in (VKOP_KSPLIT512=1) but must not silently fall back to the software
    # decode once enabled.
    vkop_shader_variant(buffer_matmul_ksplit512_w8_wide_fp8_hw_fp16
                        buffer/matmul -DFP16 -DW8_WIDE -DKSPLIT512 -DFP8_HW)
    vkop_shader_variant(buffer_matmul_ksplit512_w8_wide_bda_fp8_hw_fp16
                        buffer/matmul
                        -DFP16 -DW8_WIDE -DKSPLIT512 -DBDA -DFP8_HW)
endif()
# Pack pass: two direct addresses fit in the PC, so no table is needed.
vkop_shader_variant(buffer_matmul_pack_bda buffer/matmul_pack -DBDA)
# Primary coop pipelines: one table descriptor, same 128 B PC.
vkop_shader_variant(buffer_matmul_coop_bda buffer/matmul_coop -DBDA)
vkop_shader_variant(buffer_matmul_coop16_bda buffer/matmul_coop16 -DBDA)
# Alternative coop tile geometries (same table layout).
vkop_shader_variant(buffer_matmul_coop16_16x8x16_bda buffer/matmul_coop16
                    -DCM=16 -DCN=8 -DCK=16 -DBDA)
vkop_shader_variant(buffer_matmul_coop_fp16_8x8x16_bm8bn64_bda buffer/matmul_coop
                    -DCM=8 -DCN=8 -DCK=16 -DSG_M=1 -DSG_N=8 -DBDA)
vkop_shader_variant(buffer_matmul_coop_fp16_8x8x16_bm32bn16_bda buffer/matmul_coop
                    -DCM=8 -DCN=8 -DCK=16 -DSG_M=4 -DSG_N=2 -DBDA)
vkop_shader_variant(buffer_matmul_coop_fp16_8x8x16_bm64bn8_bda buffer/matmul_coop
                    -DCM=8 -DCN=8 -DCK=16 -DSG_M=8 -DSG_N=1 -DBDA)

# Larger 16x16x16 coopmat workgroups (MatMulBuffer::coop_geo_pipeline): the same
# matmul_coop16.comp with more accumulator tiles per subgroup, so one 256-thread
# workgroup covers 64x64 or 64x128 instead of 32x64. The host picks one from the
# shape, the base build stays the 32x64 module, and an archive converted before
# a key existed just keeps the base (resolve_spv throws by name, the op records
# the miss and falls back).
vkop_shader_variant(buffer_matmul_coop16_2x1 buffer/matmul_coop16
                    -DACC_M=2 -DACC_N=1)
vkop_shader_variant(buffer_matmul_coop16_2x2 buffer/matmul_coop16
                    -DACC_M=2 -DACC_N=2)
# BDA table builds of the two ACC geometries.
vkop_shader_variant(buffer_matmul_coop16_2x1_bda buffer/matmul_coop16
                    -DACC_M=2 -DACC_N=1 -DBDA)
vkop_shader_variant(buffer_matmul_coop16_2x2_bda buffer/matmul_coop16
                    -DACC_M=2 -DACC_N=2 -DBDA)

# The cooperative-matrix tile shapes the runtime can probe for. matmul_coop16
# carries the fp16 f32-accumulate shapes (16x16x16 is the default build above),
# matmul_coop_i8 the SINT8 ones (8x8x32 is its default build). The host picks
# the largest listed shape, so the smaller ones only matter on a device that
# lists no larger one -- but an unlisted module is never dispatched, and a
# missing key just drops to the next shape, so every listed shape needs bytes.
# ARL's 8x8x16 tile is NOT built from this source: matmul_coop.comp already
# implements it and measured 10-36% faster (one 8x8 MMA per K tile gives the
# pipelined staging nothing to hide behind).
vkop_shader_variant(buffer_matmul_coop16_16x8x16 buffer/matmul_coop16
                    -DCM=16 -DCN=8 -DCK=16)
# The 16x8x8 variant (CK=8) was also registered but it is dead: 16x8x16
# already maps to BM=32/BN=32 and the host's alt-geo lookup keys on bm/bn
# only, so CK=16 always wins. Measured on Ada and it never outperformed
# CK=16 either, so drop it entirely instead of leaving a ghost entry in the
# archive.
vkop_shader_variant(buffer_matmul_coop_i8_16x16x32 buffer/matmul_coop_i8
                    -DCM=16 -DCN=16 -DCK=32)
vkop_shader_variant(buffer_matmul_coop_i8_16x8x32 buffer/matmul_coop_i8
                    -DCM=16 -DCN=8 -DCK=32)

# UINT8→UINT32 variant of matmul_coop_i8.comp. ARL lists 8x8x32 UINT8 alongside
# SINT8 at the same tile. Ada's coopmatProps enumerate UINT8 at 16x16x32 /
# 16x8x32 — probe path (supports_coopmat_uint8_tile) picks them on Ada and the
# 8x8x32 entry stays filtered out there. Not measured on Ada yet.
# IS_SIGNED=0 flips both cooperative-matrix types (int8_t→uint8_t, int→uint)
# and the runtime quantize clamp (±127→0..255).
vkop_shader_variant(buffer_matmul_coop_u8 buffer/matmul_coop_i8
                    -DIS_SIGNED=0)
# Ada enumerates UINT8→UINT32 at 16x16x32 and 16x8x32 (ARL has neither).
# ARL's probe naturally filters them; Ada picks them up when available.
vkop_shader_variant(buffer_matmul_coop_u8_16x16x32 buffer/matmul_coop_i8
                    -DIS_SIGNED=0 -DCM=16 -DCN=16 -DCK=32)
vkop_shader_variant(buffer_matmul_coop_u8_16x8x32 buffer/matmul_coop_i8
                    -DIS_SIGNED=0 -DCM=16 -DCN=8 -DCK=32)

# BDA address-table builds of every SINT8/UINT8 coop module.
vkop_shader_variant(buffer_matmul_coop_i8_bda buffer/matmul_coop_i8 -DBDA)
vkop_shader_variant(buffer_matmul_coop_i8_16x16x32_bda buffer/matmul_coop_i8
                    -DCM=16 -DCN=16 -DCK=32 -DBDA)
vkop_shader_variant(buffer_matmul_coop_i8_16x8x32_bda buffer/matmul_coop_i8
                    -DCM=16 -DCN=8 -DCK=32 -DBDA)
vkop_shader_variant(buffer_matmul_coop_u8_bda buffer/matmul_coop_i8
                    -DIS_SIGNED=0 -DBDA)
vkop_shader_variant(buffer_matmul_coop_u8_16x16x32_bda buffer/matmul_coop_i8
                    -DIS_SIGNED=0 -DCM=16 -DCN=16 -DCK=32 -DBDA)
vkop_shader_variant(buffer_matmul_coop_u8_16x8x32_bda buffer/matmul_coop_i8
                    -DIS_SIGNED=0 -DCM=16 -DCN=8 -DCK=32 -DBDA)

# coopmat + split-K hybrid: three -D entry points per source (DEFAULT = no
# -D, SPLITK_DEPOSIT, SPLITK_FOLD). Runtime picks SPLITK_DEPOSIT for the
# main pass (fp32 partials to scratch) and SPLITK_FOLD for the reduce pass
# (fp32-sums partials, packs half2 directly to binding 0). SPLITK_DEPOSIT
# reads uPC.ksplit as coop_split; SPLITK_FOLD reads it the same way.
# matmul_coop16.comp never implemented the SPLITK_* paths, so the old
# buffer_matmul_coop16_splitk_{deposit,fold} entries were byte-identical
# duplicates of the default 16x16x16 module; they were removed rather than
# kept as a misleading alias. Host guards keep 2x2/2x1/CN=8 variants, odd N,
# and K-too-shallow shapes on the DEFAULT path regardless.
vkop_shader_variant(buffer_matmul_coop_splitk_deposit buffer/matmul_coop
                    -DSPLITK_DEPOSIT)
vkop_shader_variant(buffer_matmul_coop_splitk_fold buffer/matmul_coop
                    -DSPLITK_FOLD)

# BDA builds of the dead-but-shipped routes. No host path dispatches them
# today (the routes are disabled), so they are validated by
# tests/DeadShaderBdaProbe.cpp with a raw table dispatch.
vkop_shader_variant(buffer_matmul_coop_splitk_deposit_bda buffer/matmul_coop
                    -DSPLITK_DEPOSIT -DBDA)
vkop_shader_variant(buffer_matmul_coop_splitk_fold_bda buffer/matmul_coop
                    -DSPLITK_FOLD -DBDA)

# SPLITK_INWG: split-K inside one workgroup, no global scratch, single
# dispatch. coop_split (uPC.ksplit, 1..8) serial passes over K-range slices,
# cross-pass reduction in shared memory. Both tile sizes ship a key; the host
# still treats the route as unavailable (KernelSelect's kCoopmatInwg retry),
# so the bytes are probe/bench-only until the dispatch branch lands. Host
# guard: ACC_M==ACC_N==1 + N even + K/cs >= 2*K_BLOCK. BDA twins match the
# descriptor keys one-for-one so the dead-route table dispatch in
# tests/DeadShaderBdaProbe.cpp can A/B them.
vkop_shader_variant(buffer_matmul_coop_inwg buffer/matmul_coop
                    -DSPLITK_INWG)
vkop_shader_variant(buffer_matmul_coop_inwg_bda buffer/matmul_coop
                    -DSPLITK_INWG -DBDA)
vkop_shader_variant(buffer_matmul_coop16_inwg buffer/matmul_coop16_inwg
                    -DSPLITK_INWG)
vkop_shader_variant(buffer_matmul_coop16_inwg_bda buffer/matmul_coop16_inwg
                    -DSPLITK_INWG -DBDA)

# ARL fp16 workgroup BM/BN variants over macroized matmul_coop.comp.
# All share the same MMA tile (CM=CN=CK=8/8/16); only SG_M/SG_N/BM/BN
# change, giving 4 ways to pack 8 subgroups x 32 threads = 256 into
# local_size_x=16, local_size_y=16. Host picks via MatmulPlan::bm/bn from
# the smart tile selector. Default BM=16/BN=32 (SG_M=2/SG_N=4) is the
# no -D build keyed as buffer_matmul_coop above.
vkop_shader_variant(buffer_matmul_coop_fp16_8x8x16_bm8bn64 buffer/matmul_coop
                    -DCM=8 -DCN=8 -DCK=16 -DSG_M=1 -DSG_N=8)
vkop_shader_variant(buffer_matmul_coop_fp16_8x8x16_bm32bn16 buffer/matmul_coop
                    -DCM=8 -DCN=8 -DCK=16 -DSG_M=4 -DSG_N=2)
vkop_shader_variant(buffer_matmul_coop_fp16_8x8x16_bm64bn8 buffer/matmul_coop
                    -DCM=8 -DCN=8 -DCK=16 -DSG_M=8 -DSG_N=1)

# SK_LASTBLOCK: serial in-workgroup split-K (single dispatch, grid.z = batch).
# Per-cs precompiled variants: SK_CS baked in at compile-time avoids glslc's
# register-allocator bug with runtime ints near coopMat accumulators.
# SK_CS=2,4,8 with SK_KFLOOR=256; runtime selects by cs. Half2 variants are
# byte-identical to fp32 (ENABLE_HALF2_EPI is dead for this path) so only
# fp32 registrations kept.
vkop_shader_variant(buffer_matmul_coop16_sklast_fp32 buffer/matmul_coop16_sklast
                    -DSK_LASTBLOCK)
vkop_shader_variant(buffer_matmul_coop16_sklast_fp32_cs2 buffer/matmul_coop16_sklast
                    -DSK_LASTBLOCK -DSK_CS=2 -DSK_KFLOOR=256)
vkop_shader_variant(buffer_matmul_coop16_sklast_fp32_cs4 buffer/matmul_coop16_sklast
                    -DSK_LASTBLOCK -DSK_CS=4 -DSK_KFLOOR=256)
vkop_shader_variant(buffer_matmul_coop16_sklast_fp32_cs8 buffer/matmul_coop16_sklast
                    -DSK_LASTBLOCK -DSK_CS=8 -DSK_KFLOOR=256)
# BDA table builds of every cs variant (Ada-only route; the host keeps the
# descriptor modules as the per-dispatch fallback).
vkop_shader_variant(buffer_matmul_coop16_sklast_fp32_bda buffer/matmul_coop16_sklast
                    -DSK_LASTBLOCK -DBDA)
vkop_shader_variant(buffer_matmul_coop16_sklast_fp32_cs2_bda buffer/matmul_coop16_sklast
                    -DSK_LASTBLOCK -DSK_CS=2 -DSK_KFLOOR=256 -DBDA)
vkop_shader_variant(buffer_matmul_coop16_sklast_fp32_cs4_bda buffer/matmul_coop16_sklast
                    -DSK_LASTBLOCK -DSK_CS=4 -DSK_KFLOOR=256 -DBDA)
vkop_shader_variant(buffer_matmul_coop16_sklast_fp32_cs8_bda buffer/matmul_coop16_sklast
                    -DSK_LASTBLOCK -DSK_CS=8 -DSK_KFLOOR=256 -DBDA)

# Subgroup-arithmetic reductions (_sg) for the ops whose workgroup reduce sits
# in a hot loop or on every transformer block: attention's per-tile max/sum,
# the norm passes and softmax. -DSUBGROUP swaps the 256-slot shared-memory
# tree for subgroupAdd/subgroupMax plus one fold of the per-subgroup partials
# (see shaders/common/subgroup_reduce.comp). The base builds keep the tree and
# stay the fallback; the host prefers <key>_sg only when the device advertises
# VK_SUBGROUP_FEATURE_ARITHMETIC_BIT and the archive carries the key.
vkop_shader_variant(buffer_attention_sg buffer/attention -DSUBGROUP)
vkop_shader_variant(buffer_attention_fp16_sg buffer/attention -DFP16 -DSUBGROUP)
vkop_shader_variant(buffer_layernorm_sg buffer/layernorm -DSUBGROUP)
vkop_shader_variant(buffer_layernorm_fp16_sg buffer/layernorm -DFP16 -DSUBGROUP)
vkop_shader_variant(buffer_rmsnorm_sg buffer/rmsnorm -DSUBGROUP)
vkop_shader_variant(buffer_rmsnorm_fp16_sg buffer/rmsnorm -DFP16 -DSUBGROUP)
vkop_shader_variant(buffer_softmax_sg buffer/softmax -DSUBGROUP)
vkop_shader_variant(buffer_softmax_fp16_sg buffer/softmax -DFP16 -DSUBGROUP)
# MatMul's fused attention row softmax (matmul16_softmax) reduces each row with
# the same two trees; the -DSUBGROUP build is the base module with only that
# function's reduce swapped, so the host may prefer it for every MatMul.
vkop_shader_variant(buffer_matmul_sg buffer/matmul -DSUBGROUP)
vkop_shader_variant(buffer_matmul_fp16_sg buffer/matmul -DFP16 -DSUBGROUP)
# BDA builds of every subgroup variant. The runtime picks <key>_sg whenever
# the device advertises subgroup arithmetic and the archive has it, so the BDA
# path must offer the same reduction -- otherwise VKOP_BDA silently downgrades
# attention/norm/softmax/matmul to the shared-memory tree.
vkop_shader_variant(buffer_attention_sg_bda buffer/attention -DSUBGROUP -DBDA)
vkop_shader_variant(buffer_attention_fp16_sg_bda buffer/attention
                    -DFP16 -DSUBGROUP -DBDA)
vkop_shader_variant(buffer_layernorm_sg_bda buffer/layernorm -DSUBGROUP -DBDA)
vkop_shader_variant(buffer_layernorm_fp16_sg_bda buffer/layernorm
                    -DFP16 -DSUBGROUP -DBDA)
vkop_shader_variant(buffer_rmsnorm_sg_bda buffer/rmsnorm -DSUBGROUP -DBDA)
vkop_shader_variant(buffer_rmsnorm_fp16_sg_bda buffer/rmsnorm
                    -DFP16 -DSUBGROUP -DBDA)
vkop_shader_variant(buffer_softmax_sg_bda buffer/softmax -DSUBGROUP -DBDA)
vkop_shader_variant(buffer_softmax_fp16_sg_bda buffer/softmax
                    -DFP16 -DSUBGROUP -DBDA)
vkop_shader_variant(buffer_matmul_sg_bda buffer/matmul -DSUBGROUP -DBDA)
vkop_shader_variant(buffer_matmul_fp16_sg_bda buffer/matmul
                    -DFP16 -DSUBGROUP -DBDA)
# Image LayerNorm's three normalize shapes (W / HW / CHW) all fold sum and
# sum-of-squares through the vec4 tree; the fp32 module is the only build.
vkop_shader_variant(image_layernorm_sg image/layernorm -DSUBGROUP)

# ---------------------------------------------------------------------------
# SPIR-V external archive (vkop_shaders.bin).
#
# Every .spv produced above is packed into a single archive. It is the source
# onnx2vkop filters a model's own subset from (--master-archive) and the file
# the dev binaries read through include/dev_shaders.hpp. Keys inside the
# archive are the folder-derived names (buffer_add, image_add_fp16, ...), which
# are exactly the keys shader_key_for returns, so a lookup needs to know
# nothing about how CMake produced it.
# ---------------------------------------------------------------------------
set(ARCHIVE_ARGS "")
set(ARCHIVE_DEPS "")
foreach(SHADER ${SHADER_FILES})
    file(RELATIVE_PATH SHADER_REL ${SHADER_DIR} ${SHADER})
    string(REPLACE "/" "_" SYM_BASE "${SHADER_REL}")
    get_filename_component(SYM_BASE "${SYM_BASE}" NAME_WE)
    list(APPEND ARCHIVE_ARGS --spv ${SYM_BASE}:${SPIRV_DIR}/${SYM_BASE}.spv)
    list(APPEND ARCHIVE_DEPS ${SPIRV_DIR}/${SYM_BASE}.spv)
endforeach()
foreach(DUAL_NAME ${DUAL_FP16_SHADERS})
    string(REPLACE "/" "_" DUAL_BASE "${DUAL_NAME}")
    list(APPEND ARCHIVE_ARGS
         --spv ${DUAL_BASE}_fp16:${SPIRV_DIR}/${DUAL_BASE}_fp16.spv)
    list(APPEND ARCHIVE_DEPS ${SPIRV_DIR}/${DUAL_BASE}_fp16.spv)
endforeach()
# The -DW8_WIDE build above is neither its own .comp nor a DUAL_FP16_SHADERS
# entry (it is a second define over buffer/matmul.comp), so neither loop picks it
# up. Without this key the archive would lack it and MatMulBuffer::
# w8_wide_pipeline would get a zero-length stub, losing the wide GEMV even
# on the devices whose driver compiles it fine.
list(APPEND ARCHIVE_ARGS
     --spv buffer_matmul_w8_wide_fp16:${SPIRV_FILE_W8WIDE})
list(APPEND ARCHIVE_DEPS ${SPIRV_FILE_W8WIDE})
# Same story for the two 512-lane split-K builds above.
list(APPEND ARCHIVE_ARGS
     --spv buffer_matmul_ksplit512_fp16:${SPIRV_FILE_KS512}
     --spv buffer_matmul_ksplit512_w8_wide_fp16:${SPIRV_FILE_KS512_W8WIDE})
list(APPEND ARCHIVE_DEPS ${SPIRV_FILE_KS512} ${SPIRV_FILE_KS512_W8WIDE})
# The fp8 hardware-decode modules, when this glslc could build them. Absent
# keys are the normal state on an old shaderc, and the host falls back to the
# software decode (MatMulBuffer::f8_hw_pipeline resolves to nullptr).
if(VKOP_GLSLC_HAS_FLOAT8)
    list(APPEND ARCHIVE_ARGS
         --spv buffer_matmul_fp8_hw_fp16:${SPIRV_FILE_F8}
         --spv buffer_matmul_w8_wide_fp8_hw_fp16:${SPIRV_FILE_F8_WIDE}
         --spv buffer_conv2d_fp8_hw:${SPIRV_FILE_CONV_F8}
         --spv buffer_conv2d_fp8_hw_fp16:${SPIRV_FILE_CONV_F8_FP16})
    list(APPEND ARCHIVE_DEPS ${SPIRV_FILE_F8} ${SPIRV_FILE_F8_WIDE}
         ${SPIRV_FILE_CONV_F8} ${SPIRV_FILE_CONV_F8_FP16})
endif()
# And the two tile-split-K modules above: same .comp, two defines, so the
# SHADER_FILES loop cannot name them either.
list(APPEND ARCHIVE_ARGS
     --spv buffer_matmul_tiled_ks_split_fp16:${SPIRV_FILE_KS_SPLIT}
     --spv buffer_matmul_tiled_ks_fold_fp16:${SPIRV_FILE_KS_FOLD})
list(APPEND ARCHIVE_DEPS ${SPIRV_FILE_KS_SPLIT} ${SPIRV_FILE_KS_FOLD})
# And the SwiGLU down fold.
list(APPEND ARCHIVE_ARGS
     --spv buffer_matmul_down_act_fp16:${SPIRV_FILE_DOWN_ACT})
list(APPEND ARCHIVE_DEPS ${SPIRV_FILE_DOWN_ACT})
# And for every vkop_shader_variant build: the macro paired each key with its
# .spv, so no per-key line is needed here.
list(LENGTH VARIANT_SPV_KEYS VARIANT_COUNT)
if(VARIANT_COUNT GREATER 0)
    math(EXPR VARIANT_LAST "${VARIANT_COUNT} - 1")
    foreach(VARIANT_INDEX RANGE ${VARIANT_LAST})
        list(GET VARIANT_SPV_KEYS ${VARIANT_INDEX} VARIANT_KEY)
        list(GET VARIANT_SPV_FILES ${VARIANT_INDEX} VARIANT_FILE)
        list(APPEND ARCHIVE_ARGS --spv ${VARIANT_KEY}:${VARIANT_FILE})
        list(APPEND ARCHIVE_DEPS ${VARIANT_FILE})
    endforeach()
endif()

set(VKOP_SHADER_ARCHIVE_FILE ${CMAKE_CURRENT_BINARY_DIR}/vkop_shaders.bin)
# The --spv list outgrows CreateProcess's 32767-char command line, so on Windows
# the packer reads the same arguments from a response file, one per line.
if(WIN32)
    set(ARCHIVE_RSP "${CMAKE_CURRENT_BINARY_DIR}/vkop_shaders.rsp")
    string(REPLACE ";" "\n" ARCHIVE_RSP_ARGS "${ARCHIVE_ARGS}")
    file(WRITE "${ARCHIVE_RSP}"
         "--output\n${VKOP_SHADER_ARCHIVE_FILE}\n${ARCHIVE_RSP_ARGS}\n")
    set(ARCHIVE_CLI_ARGS "@${ARCHIVE_RSP}")
else()
    set(ARCHIVE_CLI_ARGS --output "${VKOP_SHADER_ARCHIVE_FILE}" ${ARCHIVE_ARGS})
endif()
add_custom_command(
    OUTPUT ${VKOP_SHADER_ARCHIVE_FILE}
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tools/spv_to_archive.py
        ${ARCHIVE_CLI_ARGS}
    DEPENDS ${ARCHIVE_DEPS} ${CMAKE_SOURCE_DIR}/tools/spv_to_archive.py
    COMMENT "Packing SPIR-V shaders into vkop_shaders.bin archive"
)
add_custom_target(shader_archive ALL DEPENDS ${VKOP_SHADER_ARCHIVE_FILE})
add_dependencies(shader_archive compile_shaders)


# Add executable
add_library(vkop ${SRC_FILES})
if(MSVC)
    # The feature descriptors in VulkanDevice.cpp are built with designated
    # initializers; MSVC only accepts those under /std:c++20.
    target_compile_features(vkop PUBLIC cxx_std_20)
endif()

# Compiling the SPIR-V is not a dependency of the library (no shader bytes are
# linked in), but the archive below consumes it, and shader_archive is ALL.
add_custom_target(compile_shaders DEPENDS ${SPIRV_OUTPUTS})
add_dependencies(vkop compile_shaders)

# FlatBuffers: generated schema header + vendored runtime headers live under
# model/generated. SYSTEM so -Wall/-Wextra/-Werror skip third-party code.
target_include_directories(vkop SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/model/generated)
target_link_libraries(vkop PRIVATE ${CMAKE_DL_LIBS})

add_subdirectory(model)

# Formatting is a manual `make format-code`, never a build dependency. The
# target has no OUTPUT, so as a vkop dependency it re-ran on every build and
# -i rewrote files in place, bumping mtimes and forcing needless recompiles.
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
option(USE_VMA "Enable usage of Vulkan Memory Allocator" OFF)
option(ENABLE_EXAMPLE "Enable example" ON)
option(ENABLE_BENCHMARK "Enable benchmark" ON)
option(ENABLE_TESTS "Enable testing" ON)
option(ENABLE_CUDA_LAUNCH "Enable cuda kernel launch" OFF)
option(USE_PLATFORM_WAYLAND "Enable Wayland" OFF)
# Auto-enable XCB and XLIB on Linux — virtually all desktop systems have X11
# and Vulkan loader enumerates VK_KHR_xcb_surface / VK_KHR_xlib_surface.
# GLFW on Linux picks XCB (see glfwGetRequiredInstanceExtensions output).
if(UNIX AND NOT APPLE)
    option(USE_PLATFORM_XCB "Enable XCB" ON)
    option(USE_PLATFORM_XLIB "Enable XLIB" ON)
else()
    option(USE_PLATFORM_XCB "Enable XCB" OFF)
    option(USE_PLATFORM_XLIB "Enable XLIB" OFF)
endif()
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

# vkop_gpu_top -- rootless GPU utilization monitor for profiling runs. Reads
# cumulative per-client busy time from whatever the OS already accounts for
# without privileges: /proc/<pid>/fdinfo plus i915 GT sysfs on Linux (no
# perf_event, no root), the AGX GPU-client nodes in the IORegistry on macOS (no
# powermetrics, no root). In the style of intel_gpu_top / nvtop.
# Deliberately links nothing but libc (plus IOKit on macOS), so it builds and
# runs independently of the runtime and of the GPU stack.
add_executable(vkop_gpu_top tools/vkop_gpu_top.cpp)
if (APPLE)
    target_sources(vkop_gpu_top PRIVATE tools/gpu_top_darwin.cpp)
    target_link_libraries(vkop_gpu_top PRIVATE "-framework IOKit"
                                               "-framework CoreFoundation")
else()
    target_sources(vkop_gpu_top PRIVATE tools/gpu_top_linux.cpp)
endif()

if (ENABLE_TESTS)
enable_testing()
add_subdirectory(tests)
endif()

# LLM conversational driver (Qwen3-VL, buffer backend). Text in → text out.
# Links vkop + vload (graph runtime + model loader) and the standalone Qwen
# tokenizer (BBPE encode/decode + chat template). The tokenizer ships its own
# CMake project under llm/tokenizer/; pulled in as a subdirectory its project()
# is locally scoped and enable_testing() is a no-op (the top level already
# called it), so its tokenizer_test joins the top-level ctest cleanly.
# vulkan is resolved at runtime via dlopen inside libvkop (VulkanLib.cpp), so
# no -lvulkan link is needed here.
option(ENABLE_LLM_CHAT "Build the LLM conversational driver (llm_chat)" ON)
if (ENABLE_LLM_CHAT)
    add_subdirectory(llm/tokenizer llm_tokenizer)
    add_executable(llm_chat llm/exporter/driver/llm_chat.cpp llm/exporter/driver/conversation.cc)
    target_link_libraries(llm_chat PRIVATE vkop vload tokenizer)
    target_include_directories(llm_chat PRIVATE
        ${CMAKE_SOURCE_DIR}/llm/tokenizer)
    # GPU argmax rides the engine: llm_chat registers a TopK(k=1) post-process
    # op over the logits output (Runtime::RegisterPostProcess), so the kernel
    # ships in the model's own archive (onnx2vkop --extra-shaders TopK) and
    # there is nothing to embed in the driver. The reference two-stage reduce
    # shader stays at shaders/buffer/argmax.comp (FP16 macro picks fp16/fp32
    # logits), excluded from the engine archive glob — resurrect it only if
    # TopK k=1 ever loses to it.
    find_package(Threads REQUIRED)
    target_link_libraries(llm_chat PRIVATE Threads::Threads)
    # Ensure utf8proc is linked (tokenizer's PUBLIC dependency may not propagate in some configs).
    # macOS: utf8proc is typically installed via Homebrew.
    if(APPLE)
        execute_process(COMMAND brew --prefix utf8proc OUTPUT_VARIABLE BREW_UTF8_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(BREW_UTF8_PREFIX)
            target_link_directories(llm_chat PRIVATE ${BREW_UTF8_PREFIX}/lib)
            target_include_directories(llm_chat PRIVATE ${BREW_UTF8_PREFIX}/include)
        else()
            target_link_directories(llm_chat PRIVATE /opt/homebrew/lib)
            target_include_directories(llm_chat PRIVATE /opt/homebrew/include)
        endif()
    endif()
    # MSVC 下 tokenizer 目标已经把 utf8proc.lib 的绝对路径 PUBLIC 传下来，再链一次
    # 裸名字只会在链接期报 LNK1181（Windows 没有系统 utf8proc 搜索路径）。
    if(NOT MSVC)
        target_link_libraries(llm_chat PRIVATE utf8proc)
    endif()

    # Line-protocol inference server: the HTTP/Anthropic/SSE layer lives in
    # Python (llm/server), which spawns this binary and talks role/text pairs
    # over two inherited fds. Same numerical path as llm_chat, minus the
    # performance layers (ring KV, chunked prefill, zero-copy decode inputs,
    # GPU argmax) that a correctness-first server does not need yet.
    add_executable(llm_serve llm/exporter/driver/llm_serve.cpp)
    target_link_libraries(llm_serve PRIVATE vkop vload tokenizer)
    target_include_directories(llm_serve PRIVATE
        ${CMAKE_SOURCE_DIR}/llm/tokenizer)
    target_link_libraries(llm_serve PRIVATE Threads::Threads)
    if(NOT MSVC)
        target_link_libraries(llm_serve PRIVATE utf8proc)
    endif()

    # Visual encoder probe: runs visual.vkopbin standalone to verify the visual
    # graph (incl. the patch_embed Conv3D) produces ORT-aligned outputs.
    add_executable(visual_probe llm/exporter/driver/visual_probe.cpp)
    target_link_libraries(visual_probe PRIVATE vkop vload)

    # MoE prefill GEMM M-sweep: is MatMul time linear in M, or is a fixed
    # per-dispatch / occupancy cost dominating?  Grouped (per-expert) GEMM can
    # only pay in the first case.  Standalone so it can be cross-compiled and
    # shipped to a benchmark host that has no compiler.
    add_executable(matmul_m_sweep llm/exporter/driver/matmul_m_sweep.cpp)
    target_link_libraries(matmul_m_sweep PRIVATE vkop vload)
    # libvkop ships no SPIR-V, so a binary that builds operators without a model
    # has to be handed the build tree's archive; see include/dev_shaders.hpp.
    add_dependencies(matmul_m_sweep shader_archive)
    target_compile_definitions(matmul_m_sweep PRIVATE
        "VKOP_DEV_SHADER_ARCHIVE=\"${VKOP_SHADER_ARCHIVE_FILE}\"")

    # qwen3_5 graph probe: loads an exported hybrid graph and checks that
    # every op resolves a kernel and no output goes non-finite. llm_chat's
    # consecutive-KV-name assumption does not hold for a model where only the
    # full_attention layers carry a cache, so this is the load-and-run check.
    add_executable(qwen35_probe llm/exporter/driver/qwen35_probe.cpp)
    target_link_libraries(qwen35_probe PRIVATE vkop vload)

    # KV-cache probe: runs a minimal attention+cache graph (see
    # /tmp/gen_cache_graph.py) for several rounds so the Attention op's cache
    # form is actually executed. VKOP_NO_FUSIONS=fuse_kv_cache_append on the
    # same binary is the rotation-vs-cache-form A/B.
    add_executable(cache_probe llm/exporter/driver/cache_probe.cpp)
    target_link_libraries(cache_probe PRIVATE vkop vload)

    # get_rope_index probe: verifies the C++ MRoPE position-id port against a
    # HF reference dump (dump_rope_ref.py). Header-only logic; links vkop only
    # because image_preproc.hpp -> core/Tensor.hpp pulls in VulkanLib symbols.
    add_executable(rope_probe llm/exporter/driver/rope_probe.cpp)
    target_link_libraries(rope_probe PRIVATE vkop vload)

    # npy-feed LLM driver: loads <dump_dir>/roundN/*.npy inputs, runs the model,
    # and compares 'logits' against reference_logits.npy (ORT ground truth).
    # Used for Qwen2/Qwen3 e2e correctness checks (fp16, fp8 KV, QDQ).
    add_executable(llm_driver llm/exporter/driver/llm_driver.cpp)
    target_link_libraries(llm_driver PRIVATE vkop vload)
endif()

# Image generation driver for Qwen-Image-2.1
option(ENABLE_IMAGE_GEN "Build the image generation driver (image_gen)" ON)
if (ENABLE_IMAGE_GEN)
    # image_gen tokenizes the prompt itself (raw ChatML template -> ids -> text
    # tower), so it needs the same tokenizer library llm_chat uses. The llm
    # block normally adds the subdirectory already; only add it a second time
    # when that option is off (CMake rejects a repeated add_subdirectory).
    if (NOT ENABLE_LLM_CHAT)
        add_subdirectory(llm/tokenizer llm_tokenizer)
    endif()
    add_executable(image_gen image/exporter/image_gen.cpp)
    target_link_libraries(image_gen PRIVATE vkop vload tokenizer)
    target_include_directories(image_gen PRIVATE
        ${CMAKE_SOURCE_DIR}/llm/tokenizer)
    add_executable(vae_gen image/exporter/vae_gen.cpp)
    target_link_libraries(vae_gen PRIVATE vkop vload)
    add_executable(te_gen image/exporter/te_gen.cpp)
    target_link_libraries(te_gen PRIVATE vkop vload)
endif()
