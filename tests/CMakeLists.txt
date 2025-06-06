

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

add_executable(SubTest SubTest.cpp)
target_link_libraries(SubTest -Wl,--whole-archive vkop -Wl,--no-whole-archive)

add_test(NAME SubTest COMMAND SubTest)


add_executable(MulTest MulTest.cpp)
target_link_libraries(MulTest -Wl,--whole-archive vkop -Wl,--no-whole-archive)

add_test(NAME MulTest COMMAND MulTest)

add_executable(DivTest DivTest.cpp)
target_link_libraries(DivTest -Wl,--whole-archive vkop -Wl,--no-whole-archive)

add_test(NAME DivTest COMMAND DivTest)