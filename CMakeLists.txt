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
        DEPENDS ${SHADER_DIR}/${DUAL_NAME}.comp
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
vkop_shader_variant(buffer_matmul_coop16_16x8x8 buffer/matmul_coop16
                    -DCM=16 -DCN=8 -DCK=8)
vkop_shader_variant(buffer_matmul_coop_i8_16x16x32 buffer/matmul_coop_i8
                    -DCM=16 -DCN=16 -DCK=32)

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
# And the two tile-split-K modules above: same .comp, two defines, so the
# SHADER_FILES loop cannot name them either.
list(APPEND ARCHIVE_ARGS
     --spv buffer_matmul_tiled_ks_split_fp16:${SPIRV_FILE_KS_SPLIT}
     --spv buffer_matmul_tiled_ks_fold_fp16:${SPIRV_FILE_KS_FOLD})
list(APPEND ARCHIVE_DEPS ${SPIRV_FILE_KS_SPLIT} ${SPIRV_FILE_KS_FOLD})
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
add_custom_command(
    OUTPUT ${VKOP_SHADER_ARCHIVE_FILE}
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tools/spv_to_archive.py
        --output ${VKOP_SHADER_ARCHIVE_FILE}
        ${ARCHIVE_ARGS}
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
option(USE_VMA "Enable usage of Vulkan Memory Allocator" OFF)
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
    target_link_libraries(llm_chat PRIVATE utf8proc)

    # Visual encoder probe: runs visual.vkopbin standalone to verify the visual
    # graph (incl. the patch_embed Conv3D) produces ORT-aligned outputs.
    add_executable(visual_probe llm/exporter/driver/visual_probe.cpp)
    target_link_libraries(visual_probe PRIVATE vkop vload)

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
