
add_executable(tokenizer_test main.cpp)

target_link_libraries(tokenizer_test PRIVATE tokenizer)

add_test(tokenizer_test tokenizer_test)