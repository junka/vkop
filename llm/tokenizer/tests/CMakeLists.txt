
add_executable(tokenizer_test main.cpp)

target_link_libraries(tokenizer_test PRIVATE tokenizer)

# 测试数据：qwen3_vl.bin（由 tokenizer_to_bin.py 生成在源码根）与
# hf_ground_truth.json（tests/ 下）。测试可执行在 build/tests/，运行时按
# 「可执行文件旁 → 源码根」顺序查找（见 main.cpp 的 find_data_file），故这里
# 用 copy_if_different 把两份数据拷到测试可执行同目录，保证 `make test` /
# ctest 从任意 cwd 运行都能找到，且源文件更新后自动重拷。
set(QWEN3_BIN_SRC "${CMAKE_SOURCE_DIR}/qwen3_vl.bin")
set(QWEN3_BIN_DST "${CMAKE_CURRENT_BINARY_DIR}/qwen3_vl.bin")
set(HF_GT_SRC     "${CMAKE_CURRENT_SOURCE_DIR}/hf_ground_truth.json")
set(HF_GT_DST     "${CMAKE_CURRENT_BINARY_DIR}/hf_ground_truth.json")

if(NOT EXISTS "${QWEN3_BIN_SRC}")
    message(WARNING
        "qwen3_vl.bin 不存在于 ${CMAKE_SOURCE_DIR}。\n"
        "  测试会失败，直到你运行 `python3 tokenizer_to_bin.py` 生成它。")
endif()

add_custom_command(
    OUTPUT  "${QWEN3_BIN_DST}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${QWEN3_BIN_SRC}" "${QWEN3_BIN_DST}"
    DEPENDS "${QWEN3_BIN_SRC}"
    VERBATIM)
add_custom_command(
    OUTPUT  "${HF_GT_DST}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${HF_GT_SRC}" "${HF_GT_DST}"
    DEPENDS "${HF_GT_SRC}"
    VERBATIM)
add_custom_target(copy_test_data ALL DEPENDS "${QWEN3_BIN_DST}" "${HF_GT_DST}")
add_dependencies(tokenizer_test copy_test_data)

add_test(NAME tokenizer_test COMMAND tokenizer_test)
