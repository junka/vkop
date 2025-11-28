

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
    BatchNorm2dTest
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
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
if(Python3_FOUND AND Python3_NumPy_FOUND)
    # Python3_INCLUDE_DIRS 可能不会直接由 FindPython3 提供，而是通过 Python3_EXECUTABLE 来获取路径
    # 下面是一个示例如何获取正确的包含路径，实际情况可能会有所不同
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
                    OUTPUT_VARIABLE PYTHON3_INCLUDE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    # 获取 NumPy 的 include 目录
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
                    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    include_directories(${PYTHON3_INCLUDE_DIR})
    include_directories(${NUMPY_INCLUDE_DIR})

    message(STATUS "Found Python3: ${Python3_VERSION} at ${PYTHON3_INCLUDE_DIR}")
    message(STATUS "Found NumPy at ${NUMPY_INCLUDE_DIR}")
else()
    if(NOT Python3_FOUND)
        message(FATAL_ERROR "Could not find Python3")
    endif()
    if(NOT Python3_NumPy_FOUND)
        message(FATAL_ERROR "Could not find NumPy for Python3")
    endif()
endif()

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    if (USE_FP16)
        target_compile_definitions(${TEST} PUBLIC FP16)
    endif()
    target_link_libraries(${TEST} -Wl,--whole-archive vkop vload -Wl,--no-whole-archive -lpthread)
    if (Python3_FOUND)
        target_link_libraries(${TEST} Python3::Python)
    endif()
    add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()

file(COPY ../model/add_model.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/model)
file(COPY ../model/add_conv_model.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/model)

target_compile_definitions(ModelTest PRIVATE
    TEST_DATA_PATH="${CMAKE_CURRENT_BINARY_DIR}/model"
)