add_executable(minimax_h3 main.cpp)
target_link_libraries(minimax_h3 PRIVATE vkop vload)
target_compile_features(minimax_h3 PRIVATE cxx_std_20)

# The text path needs the Qwen tokenizer. The top-level file adds
# llm/tokenizer BEFORE example/ (it has to: a target added later in the file is
# not visible here), so under ENABLE_LLM_CHAT the target simply exists. With it
# off nothing adds the tokenizer, and this file would have to -- but a second
# add_subdirectory of that source tree fails, because
# llm/tokenizer/tests/CMakeLists.txt declares a `copy_test_data` custom target
# that CMake refuses to define twice. So the driver builds without --prompt in
# that configuration, and says so at run time.
if (TARGET tokenizer)
    target_link_libraries(minimax_h3 PRIVATE tokenizer)
    target_include_directories(minimax_h3 PRIVATE
        ${CMAKE_SOURCE_DIR}/llm/tokenizer)
    target_compile_definitions(minimax_h3 PRIVATE H3_HAVE_TOKENIZER=1)
else ()
    message(STATUS
        "minimax_h3: tokenizer target absent (ENABLE_LLM_CHAT off); "
        "--prompt will be refused, everything else still builds")
endif ()
