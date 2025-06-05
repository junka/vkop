

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/ops)

add_executable(GridSampleTest GridSampleTest.cpp)

target_link_libraries(GridSampleTest -Wl,--whole-archive vkop -Wl,--no-whole-archive)

add_test(NAME GridSampleTest COMMAND GridSampleTest)



add_executable(Conv2dTest Conv2dTest.cpp)
target_link_libraries(Conv2dTest -Wl,--whole-archive vkop -Wl,--no-whole-archive)

add_test(NAME Conv2dTest COMMAND Conv2dTest)




add_executable(AddTest AddTest.cpp)
target_link_libraries(AddTest -Wl,--whole-archive vkop -Wl,--no-whole-archive)

add_test(NAME AddTest COMMAND AddTest)