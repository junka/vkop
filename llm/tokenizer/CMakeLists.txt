cmake_minimum_required(VERSION 3.10)
project(QwenTokenizer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# utf8proc 用于 normalizer 的 NFC 规范化，以及 pre_tokenizer 的 \p{L}/\p{N}/\s
# Unicode 类别判定。pre_tokenizer 的 GPT-2 正则（含 \s+(?!\S) 前瞻语义）改为
# 手写确定性扫描器实现，故不再依赖 re2。系统装了 dev 包时用 pkg-config 找。
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(UTF8PROC QUIET libutf8proc)
endif()
if(NOT UTF8PROC_FOUND)
    message(FATAL_ERROR "libutf8proc-dev not found. Install it (e.g. apt install libutf8proc-dev) to build the tokenizer.")
endif()

add_library(tokenizer STATIC tokenizer.cpp utf8.cpp)

# tokenizer.hpp 在公开接口里就 include 了 <utf8proc.h>，所以头文件路径必须是
# PUBLIC —— 否则下游（tests/ 与 llm_chat）在自己的目标里编不过。链接同理：
# -L 要交给消费者，静态库不会把库搜索路径带过去（macOS 上 brew 的
# /opt/homebrew/lib 不在默认搜索路径里，此前 target 链接直接失败）。
target_include_directories(tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tokenizer PUBLIC ${UTF8PROC_INCLUDE_DIRS})
target_link_libraries(tokenizer PUBLIC ${UTF8PROC_LIBRARIES})
target_link_directories(tokenizer PUBLIC ${UTF8PROC_LIBRARY_DIRS})
target_link_options(tokenizer PRIVATE ${UTF8PROC_LDFLAGS})

enable_testing()
add_subdirectory(tests)
