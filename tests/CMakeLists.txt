

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/ops)

add_executable(GridSampleTest GridSampleTest.cpp)
target_link_libraries(GridSampleTest vkop)

add_test(NAME GridSampleTest COMMAND GridSampleTest)