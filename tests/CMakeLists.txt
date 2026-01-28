

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/ops)
include_directories(${CMAKE_SOURCE_DIR}/model)

set(TESTS
    GridSampleTest
    Conv2dTest
    AddTest
    AtanTest
    SubTest
    MulTest
    DivTest
    PowTest
    ErfTest
    ReluTest
    BatchNormTest
    SoftmaxTest
    FloorTest
    MaxPoolTest
    LayerNormTest
    ResizeTest
    PReluTest
    SigmoidTest
    SoftplusTest
    ModelTest
    MatMulTest
    GemmTest
    ReduceTest
    ReshapeTest
    TransposeTest
    SliceTest
    ConcatTest
    SplitTest
    GlobalAveragePoolTest
    AveragePoolTest
    TopkTest
)

set(LIBTORCH_VERSION "2.10.0")
# Determine the appropriate libtorch URL based on the operating system
if(WIN32)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
elseif(APPLE)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-shared-with-deps-${LIBTORCH_VERSION}.zip")
else() # Linux
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
endif()
message(STATUS "Using LibTorch from: ${LIBTORCH_URL}")
set(LIBTORCH_ROOT ${CMAKE_BINARY_DIR}/libtorch)
set(LIBTORCH_DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/downloads)

# Check if libtorch is already downloaded
if(NOT EXISTS "${LIBTORCH_ROOT}")
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

# Use the downloaded libtorch
set(TORCH_LIB_DIR "${LIBTORCH_ROOT}/lib")
set(TORCH_INCLUDE_DIR "${LIBTORCH_ROOT}/include")
set(TORCH_API_INCLUDE_DIR "${LIBTORCH_ROOT}/include/torch/csrc/api/include")

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
foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    if (USE_FP16)
        target_compile_definitions(${TEST} PUBLIC FP16)
    endif()
    target_compile_options(${TEST} PRIVATE
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-function
    )
    target_link_libraries(${TEST} PRIVATE vkop vload ${TORCH_LIBRARY}
        ${TORCH_CPU_LIBRARY}
        ${C10_LIBRARY})
    target_include_directories(${TEST} PRIVATE ${TORCH_INCLUDE_DIR} ${TORCH_API_INCLUDE_DIR})
    add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()

file(COPY ../model/add_conv_model.vkopbin DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/model)

target_compile_definitions(ModelTest PRIVATE
    TEST_DATA_PATH="${CMAKE_CURRENT_BINARY_DIR}/model"
)