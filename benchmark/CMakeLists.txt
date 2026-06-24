
option(USE_GPU_POSTPROCESS "Enable GPU Postprocess" ON)
add_executable(vkbench main.cpp)
target_link_libraries(vkbench vkop vload)
