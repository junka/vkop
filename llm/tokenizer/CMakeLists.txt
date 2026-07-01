cmake_minimum_required(VERSION 3.10)
project(QwenTokenizer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tokenizer STATIC tokenizer.cpp)

target_include_directories(tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_subdirectory(tests)