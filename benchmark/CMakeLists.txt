
add_executable(vkbench main.cpp)
target_link_libraries(vkbench vkop vload)

# Generic .vkopbin benchmark driver, model-agnostic. Where vkbench above is
# CNN-specific (needs a real image + label file to run and verify top-k),
# vkop-bench takes any .vkopbin, feeds dtype-matched pseudo-random data into
# every input (with concrete shapes from --shape for dynamic dims), and
# times Run() over N iterations reporting min/p50/p90/p99/max. Correctness
# is not verified; tools/vkop_vs_ort.py is the generic check (it drives this
# binary's --load-inputs/--save-outputs against an ONNX Runtime reference),
# llm_driver the LLM-specific one. Same deployment env knobs as llm_chat:
# VK_PIPELINE_CACHE, MESA_SHADER_CACHE_DISABLE.
add_executable(vkop_bench vkop_bench.cpp)
target_link_libraries(vkop_bench PRIVATE vkop vload)
set_target_properties(vkop_bench PROPERTIES OUTPUT_NAME vkop-bench)

# Standalone MatMul microbenchmark — decode-shape fp16, buffer backend.
# Carries a per-shape correctness check (fp32 CPU reference).
add_executable(bench_matmul bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE vkop vload)
target_compile_definitions(bench_matmul PRIVATE
    VKOP_DEV_SHADER_ARCHIVE="${CMAKE_BINARY_DIR}/vkop_shaders.bin")

# Same microbenchmark against the int8 weight-only MatMul routes.
add_executable(bench_matmul_i8 bench_matmul_i8.cpp)
target_link_libraries(bench_matmul_i8 PRIVATE vkop vload)
target_compile_definitions(bench_matmul_i8 PRIVATE
    VKOP_DEV_SHADER_ARCHIVE="${CMAKE_BINARY_DIR}/vkop_shaders.bin")

