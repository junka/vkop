
add_executable(tokenizer_test main.cpp)

target_link_libraries(tokenizer_test PRIVATE tokenizer)

# 测试数据：qwen3_vl.bin（由 tokenizer_to_bin.py 生成在 llm/tokenizer/ 下）与
# hf_ground_truth.json（tests/ 下）。测试可执行在 build/tests/，运行时按
# 「可执行文件旁 → 源码根」顺序查找（见 main.cpp 的 find_data_file），故这里
# 用 copy_if_different 把两份数据拷到测试可执行同目录，保证 `make test` /
# ctest 从任意 cwd 运行都能找到，且源文件更新后自动重拷。
set(QWEN3_BIN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../qwen3_vl.bin")
set(QWEN3_BIN_DST "${CMAKE_CURRENT_BINARY_DIR}/qwen3_vl.bin")
set(HF_GT_SRC     "${CMAKE_CURRENT_SOURCE_DIR}/hf_ground_truth.json")
set(HF_GT_DST     "${CMAKE_CURRENT_BINARY_DIR}/hf_ground_truth.json")
# 第二个模型（Phi-4-mini）的同一份测试：数据存在才注册，缺了就跳过。
set(PHI_BIN_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/../phi4_mini.bin")
set(PHI_BIN_DST   "${CMAKE_CURRENT_BINARY_DIR}/phi4_mini.bin")
set(PHI_GT_SRC    "${CMAKE_CURRENT_SOURCE_DIR}/phi_ground_truth.json")
set(PHI_GT_DST    "${CMAKE_CURRENT_BINARY_DIR}/phi_ground_truth.json")

if(NOT EXISTS "${QWEN3_BIN_SRC}")
    message(WARNING
        "qwen3_vl.bin 不存在于 ${QWEN3_BIN_SRC}。\n"
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

if(EXISTS "${PHI_BIN_SRC}" AND EXISTS "${PHI_GT_SRC}")
    add_custom_command(
        OUTPUT  "${PHI_BIN_DST}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PHI_BIN_SRC}" "${PHI_BIN_DST}"
        DEPENDS "${PHI_BIN_SRC}"
        VERBATIM)
    add_custom_command(
        OUTPUT  "${PHI_GT_DST}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PHI_GT_SRC}" "${PHI_GT_DST}"
        DEPENDS "${PHI_GT_SRC}"
        VERBATIM)
    add_custom_target(copy_phi_test_data ALL
        DEPENDS "${PHI_BIN_DST}" "${PHI_GT_DST}")
    add_dependencies(tokenizer_test copy_phi_test_data)
    add_test(NAME tokenizer_test_phi
             COMMAND tokenizer_test ${PHI_BIN_DST} ${PHI_GT_DST})
endif()
