
option(USE_GPU_POSTPROCESS "Enable GPU Postprocess" ON)
add_executable(vkbench main.cpp)
target_link_libraries(vkbench vkop vload)

# Generic .vkopbin benchmark driver, model-agnostic. Where vkbench above is
# CNN-specific (needs a real image + label file to run and verify top-k),
# vkop-bench takes any .vkopbin, feeds dtype-matched pseudo-random data into
# every input (with concrete shapes from --shape for dynamic dims), and
# times Run() over N iterations reporting min/p50/p90/p99/max. Correctness
# is not verified; use llm_driver or compare_vkop_ort.py for that. Same
# deployment env knobs as llm_chat: VK_PIPELINE_CACHE,
# MESA_SHADER_CACHE_DISABLE.
add_executable(vkop_bench vkop_bench.cpp)
target_link_libraries(vkop_bench PRIVATE vkop vload)
set_target_properties(vkop_bench PROPERTIES OUTPUT_NAME vkop-bench)
