

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
    # ReduceTest
    ReshapeTest
    TransposeTest
    SliceTest
    ConcatTest
    SplitTest
    GlobalAveragePoolTest
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if (DEFINED Python3_INCLUDE_DIRS)
    include_directories(${Python3_INCLUDE_DIRS})
else()
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
                    OUTPUT_VARIABLE PYTHON3_INCLUDE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories(${PYTHON3_INCLUDE_DIR})
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import importlib,sys; spec=importlib.util.find_spec('torch'); print('FOUND' if spec else 'MISSING'); print(spec.origin if spec else '')"
    OUTPUT_VARIABLE PYTORCH_CHECK
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

string(REPLACE "\n" ";" PYTORCH_LINES "${PYTORCH_CHECK}")
list(GET PYTORCH_LINES 0 PYTORCH_FOUND)
if (NOT PYTORCH_FOUND STREQUAL "FOUND")
    message(FATAL_ERROR "Python interpreter (${Python3_EXECUTABLE}) cannot import 'torch'. Install PyTorch in that Python environment or use a different Python.")
else()
    list(GET PYTORCH_LINES 1 PYTORCH_ORIGIN)
    message(STATUS "Python 'torch' found at: ${PYTORCH_ORIGIN}")
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