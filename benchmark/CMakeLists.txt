

add_executable(vkbench main.cpp)
target_link_libraries(vkbench -Wl,--whole-archive vkop -Wl,--no-whole-archive vload -lpthread)
