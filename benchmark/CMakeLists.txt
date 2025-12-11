

add_executable(vkresnet50 resnet50v2.cpp)
target_link_libraries(vkresnet50 -Wl,--whole-archive vkop -Wl,--no-whole-archive vload -lpthread)
