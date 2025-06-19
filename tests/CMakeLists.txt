

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/ops)

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
)

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} -Wl,--whole-archive vkop -Wl,--no-whole-archive)
    add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()
