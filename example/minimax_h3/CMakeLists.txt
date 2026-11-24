add_executable(minimax_h3 main.cpp)
target_link_libraries(minimax_h3 PRIVATE vkop vload)
target_compile_features(minimax_h3 PRIVATE cxx_std_17)
