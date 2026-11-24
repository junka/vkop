add_executable(minimax_h3 main.cpp)
target_link_libraries(minimax_h3 PRIVATE vkop vload)
target_compile_features(minimax_h3 PRIVATE cxx_std_20)

# The text path needs the tokenizer. It is normally built by the ENABLE_LLM_CHAT
# block, which sits AFTER `add_subdirectory(example)` in the top-level file, so
# that target does not exist yet here. Built in a directory-scoped subdir of its
# own it is the same library under a distinct CMake target name, which avoids
# both the forward-reference and a duplicate-target error when ENABLE_LLM_CHAT
# is on. With neither available the driver still builds and falls back to
# in-vocabulary ids.
if (ENABLE_LLM_CHAT)
    target_link_libraries(minimax_h3 PRIVATE tokenizer)
    target_include_directories(minimax_h3 PRIVATE
        ${CMAKE_SOURCE_DIR}/llm/tokenizer)
    target_compile_definitions(minimax_h3 PRIVATE H3_HAVE_TOKENIZER=1)
elseif (EXISTS ${CMAKE_SOURCE_DIR}/llm/tokenizer/CMakeLists.txt)
    add_subdirectory(${CMAKE_SOURCE_DIR}/llm/tokenizer
                     ${CMAKE_BINARY_DIR}/llm_tokenizer_for_h3)
    target_link_libraries(minimax_h3 PRIVATE tokenizer)
    target_include_directories(minimax_h3 PRIVATE
        ${CMAKE_SOURCE_DIR}/llm/tokenizer)
    target_compile_definitions(minimax_h3 PRIVATE H3_HAVE_TOKENIZER=1)
endif ()
