cmake_minimum_required(VERSION 3.10)
project(QwenTokenizer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# re2 用于 pre_tokenizer 的 GPT-2 正则切分（含 \p{L}/\p{N} Unicode 属性类）。
# utf8proc 用于 normalizer 的 NFC 规范化。系统装了 dev 包时用 pkg-config 找。
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(RE2 QUIET re2)
    pkg_check_modules(UTF8PROC QUIET libutf8proc)
endif()
if(NOT RE2_FOUND)
    message(FATAL_ERROR "libre2-dev not found. Install it (e.g. apt install libre2-dev) to build the tokenizer.")
endif()
if(NOT UTF8PROC_FOUND)
    message(FATAL_ERROR "libutf8proc-dev not found. Install it (e.g. apt install libutf8proc-dev) to build the tokenizer.")
endif()

add_library(tokenizer STATIC tokenizer.cpp)

target_include_directories(tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tokenizer PRIVATE ${RE2_INCLUDE_DIRS} ${UTF8PROC_INCLUDE_DIRS})
target_link_libraries(tokenizer PUBLIC ${RE2_LIBRARIES} ${UTF8PROC_LIBRARIES})
target_link_options(tokenizer PRIVATE ${RE2_LDFLAGS} ${UTF8PROC_LDFLAGS})

enable_testing()
add_subdirectory(tests)
