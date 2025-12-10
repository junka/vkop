

add_executable(vkyolo main.cpp)
target_link_libraries(vkyolo -Wl,--whole-archive vkop -Wl,--no-whole-archive vload -lpthread)
