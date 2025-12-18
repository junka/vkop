

add_executable(vkresnet resnet.cpp)
target_link_libraries(vkresnet -Wl,--whole-archive vkop -Wl,--no-whole-archive vload -lpthread)
