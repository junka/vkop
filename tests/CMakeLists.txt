

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
    FloorTest
    MaxPoolTest
)

find_package(PythonLibs REQUIRED)
if (PYTHONLIBS_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    message(STATUS "Found PythonLibs: ${PYTHONLIBS_VERSION_STRING} at ${PYTHONLIBS_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Could not find PythonLibs")
endif()

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} -Wl,--whole-archive vkop -Wl,--no-whole-archive)
    if (PYTHONLIBS_FOUND)
        target_link_libraries(${TEST} ${PYTHON_LIBRARIES})
    endif()
    add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()
