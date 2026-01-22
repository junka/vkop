

add_executable(vkbench main.cpp)
target_link_libraries(vkbench vkop vload -lpthread)
