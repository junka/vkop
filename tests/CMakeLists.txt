
include(FetchContent)
include(GoogleTest)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")


# Try to find gtest locally first, fall back to FetchContent
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )

    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
    endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/ops)
include_directories(${CMAKE_SOURCE_DIR}/model)

set(TEST_SOURCES
    setup.cpp
    GridSampleTest.cpp
    Conv2dTest.cpp
    AddTest.cpp
    AtanTest.cpp
    SubTest.cpp
    MulTest.cpp
    DivTest.cpp
    PowTest.cpp
    ErfTest.cpp
    ReluTest.cpp
    BatchNormTest.cpp
    SoftmaxTest.cpp
    FloorTest.cpp
    MaxPoolTest.cpp
    LayerNormTest.cpp
    ResizeTest.cpp
    PReluTest.cpp
    SigmoidTest.cpp
    SoftplusTest.cpp
    MatMulTest.cpp
    GemmTest.cpp
    ReduceTest.cpp
    ReshapeTest.cpp
    TransposeTest.cpp
    SliceTest.cpp
    ConcatTest.cpp
    SplitTest.cpp
    GlobalAveragePoolTest.cpp
    AveragePoolTest.cpp
    TopkTest.cpp
    NmsTest.cpp
    GatherTest.cpp
    RangeTest.cpp
    ExpandTest.cpp
    TanhTest.cpp
    BufferBackendTest.cpp
    EmbeddingForwardTest.cpp
    EqualTest.cpp
    NonZeroTest.cpp
    ScatterElementsTest.cpp
    ShapeTest.cpp
    CastTest.cpp
    ScatterNDTest.cpp
    BufferRankTest.cpp
)

# Option: compile all tests into a single translation unit for faster builds.
# This avoids re-parsing <torch/torch.h> in each test file (~7s per file).
# When enabled, TEST_SOURCES is replaced by a single combined source.
option(MERGE_TEST_SOURCES "Merge all test sources into one TU for faster compilation" ON)

# ---- GCC version guard for macOS --------------------------------------------
# The actual libc++ ABI bridge lives in the top-level CMakeLists.txt.  We
# duplicate just the version check here so users get a clear, early error
# before libtorch is downloaded / configured.
# -----------------------------------------------------------------------------
if(APPLE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
          AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpfullversion -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GCC_VERSION VERSION_GREATER_EQUAL 15)
        message(FATAL_ERROR
            "GCC ${GCC_VERSION} + Apple libc++ is not supported. "
            "Please either:\n"
            "  1. Install GCC 14:    `brew install gcc@14` and rebuild with\n"
            "                        -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-14\n"
            "  2. Use Apple clang:   -DCMAKE_CXX_COMPILER=/usr/bin/clang++"
        )
    endif()
endif()

set(LIBTORCH_VERSION "2.10.0")
find_package(Python3 COMPONENTS Interpreter REQUIRED)
# Determine the appropriate libtorch URL based on the operating system
if(WIN32)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        message(STATUS "Detected macOS ARM64 architecture")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        message(STATUS "Detected macOS x86_64 architecture")
        # Python3 is resolved before this block, so when CMake is launched
        # from a venv these tags match the venv's interpreter and pip installs
        # into that same environment instead of attempting a system install.
        set(PYTHON_WHEEL_TAG "cp${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-cp${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
        set(TORCH_URL "https://github.com/Morton-Li/PyTorch-MacOS-Builder/releases/download/v${LIBTORCH_VERSION}/torch-${LIBTORCH_VERSION}-${PYTHON_WHEEL_TAG}-macosx_11_0_x86_64.whl")
        set(TORCHVISION_URL "https://github.com/Morton-Li/TorchVision-MacOS-Builder/releases/download/v0.28.0/torchvision-0.28.0-${PYTHON_WHEEL_TAG}-macosx_11_0_x86_64.whl")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__version__)"
            RESULT_VARIABLE TORCH_VERSION_RESULT
            OUTPUT_VARIABLE TORCH_PYTHON_VERSION
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT TORCH_VERSION_RESULT EQUAL 0 OR NOT TORCH_PYTHON_VERSION MATCHES "^2\\.10\\.")
            message(WARNING "The installed torch version (${TORCH_PYTHON_VERSION}) is incompatible. Install PyTorch and TorchVision from these repositories. Select the wheel cp3xx tag matching your Python version (for example, cp312 for Python 3.12):\n  ${TORCH_URL}\n  ${TORCHVISION_URL}")
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -m pip install --no-deps ${TORCH_URL} ${TORCHVISION_URL}
                RESULT_VARIABLE PIP_INSTALL_RESULT
                ERROR_VARIABLE PIP_INSTALL_ERROR
            )
            if(NOT PIP_INSTALL_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to install PyTorch/TorchVision: ${PIP_INSTALL_ERROR}")
            endif()
        endif()
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
            RESULT_VARIABLE TORCH_PATH_RESULT
            OUTPUT_VARIABLE LIBTORCH_ROOT
            ERROR_VARIABLE TORCH_PATH_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT TORCH_PATH_RESULT EQUAL 0 OR NOT EXISTS "${LIBTORCH_ROOT}/include" OR NOT EXISTS "${LIBTORCH_ROOT}/lib")
            message(FATAL_ERROR "The torch directory in Python site-packages must contain include and lib: ${TORCH_PATH_ERROR}")
        endif()
    else()
        message(FATAL_ERROR "Unsupported macOS architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
else() # Linux
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
endif()

if(NOT LIBTORCH_ROOT)
    message(STATUS "need to download libtorch...")
    set(LIBTORCH_ROOT ${CMAKE_BINARY_DIR}/libtorch)
endif()

# Check if libtorch is already downloaded
if(NOT EXISTS "${LIBTORCH_ROOT}")
    set(LIBTORCH_DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/downloads)
    message(STATUS "Downloading libtorch...")

    # Create download directory
    file(MAKE_DIRECTORY ${LIBTORCH_DOWNLOAD_DIR})

    # Download the libtorch archive
    set(LIBTORCH_ZIP_PATH "${LIBTORCH_DOWNLOAD_DIR}/libtorch.zip")
    file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ZIP_PATH} STATUS DOWNLOAD_STATUS)

    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_STATUS_CODE)
    if(NOT DOWNLOAD_STATUS_CODE EQUAL 0)
        list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
        message(FATAL_ERROR "Failed to download libtorch: ${ERROR_MESSAGE}")
    endif()

    # Extract the archive
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ZIP_PATH}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

message(STATUS "Using LibTorch from: ${LIBTORCH_ROOT}")
set(TORCH_LIB_DIR "${LIBTORCH_ROOT}/lib")
set(TORCH_INCLUDE_DIR "${LIBTORCH_ROOT}/include")
set(TORCH_API_INCLUDE_DIR "${LIBTORCH_ROOT}/include/torch/csrc/api/include")

set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch" CACHE PATH "Path to Torch cmake config")
find_package(Torch REQUIRED)

find_library(TORCH_LIBRARY
    NAMES torch
    PATHS ${TORCH_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)
find_library(TORCH_CPU_LIBRARY
    NAMES torch_cpu
    PATHS ${TORCH_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(C10_LIBRARY
    NAMES c10
    PATHS ${TORCH_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)
message(STATUS "Using Torch libraries: ${TORCH_LIBRARIES}")

if(MERGE_TEST_SOURCES)
    # Generate a single combined source that includes all test files.
    # This way <torch/torch.h> is parsed only once, cutting build time
    # from ~7s * N_test_files down to ~7s * 1.
    set(ALL_TESTS_COMBINED ${CMAKE_CURRENT_BINARY_DIR}/all_tests_combined.cpp)
    file(WRITE ${ALL_TESTS_COMBINED}
        "// Auto-generated - do not edit\n"
        "// Merges all test sources into one translation unit for faster compilation.\n"
    )
    foreach(SRC ${TEST_SOURCES})
        # Exclude setup.cpp since it contains main() and static definitions
        if(NOT SRC STREQUAL "setup.cpp")
            file(APPEND ${ALL_TESTS_COMBINED} "#include \"${CMAKE_CURRENT_SOURCE_DIR}/${SRC}\"\n")
        endif()
    endforeach()

    add_executable(all_tests setup.cpp ${ALL_TESTS_COMBINED})
    set_source_files_properties(${ALL_TESTS_COMBINED} PROPERTIES GENERATED TRUE)
else()
    add_executable(all_tests ${TEST_SOURCES})
endif()

if (USE_FP16)
    target_compile_definitions(all_tests PUBLIC FP16)
endif()
if(MSVC)
    target_compile_options(all_tests PRIVATE /wd4100)
else()
    target_compile_options(all_tests PRIVATE
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-dangling-reference
    )
endif()
target_link_libraries(all_tests PRIVATE vkop vload -Wl,-force_load ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    GTest::gtest GTest::gtest_main)
if(NOT GTest_FOUND)
    target_include_directories(all_tests PRIVATE
        ${googletest_SOURCE_DIR}/googletest/include)
endif()
target_include_directories(all_tests PRIVATE ${TORCH_INCLUDE_DIR} ${TORCH_API_INCLUDE_DIR})

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_LIB_DIR}/*.dll")
    add_custom_command(TARGET all_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:all_tests>
        COMMENT "Copying LibTorch DLLs to output directory"
    )
endif()

gtest_discover_tests(all_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES
        TIMEOUT 300
)

# Regenerate the test fixture from the source ONNX via the current onnx2vkop
# writer at build time, so the .vkopbin always tracks both ends of the format.
# The .onnx is the tracked source of truth; the .vkopbin is a build artifact.
# ModelTest requires the int8-quantized fixture (it reads conv.weight as int8
# + conv.weight_scale as float), hence -q int8.
set(MODEL_ONNX ${CMAKE_SOURCE_DIR}/model/add_conv_model.onnx)
set(MODEL_VKOPBIN ${CMAKE_CURRENT_BINARY_DIR}/model/add_conv_model.vkopbin)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_command(
    OUTPUT ${MODEL_VKOPBIN}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/model
    COMMAND ${Python3_EXECUTABLE} -m onnx2vkop.cli
            -i ${MODEL_ONNX}
            -o ${MODEL_VKOPBIN}
            -q int8
    DEPENDS ${MODEL_ONNX}
    COMMENT "Converting add_conv_model.onnx -> add_conv_model.vkopbin (int8)"
    VERBATIM
)

add_executable(ModelTest ModelTest.cpp ${MODEL_VKOPBIN})
target_link_libraries(ModelTest PRIVATE vkop vload)
target_compile_definitions(ModelTest PRIVATE
    TEST_DATA_PATH="${CMAKE_CURRENT_BINARY_DIR}/model"
)
add_test(ModelTest ModelTest)