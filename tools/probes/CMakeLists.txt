# vkop op probes -- one binary, dispatched by name.
#
# These probes used to be 26 executables in tests/, each recompiling the whole
# vkop header prefix (~5.5s) on every header edit for ~2.8s of actual probe
# code. They are now one target, and the probe sources are compiled in a few
# merged translation units rather than one TU each (see PROBE_BUCKETS).
#
# VKOP_PREBUILT_PROBES:
#   OFF (default)  probe sources are merged into PROBE_BUCKETS TUs, so a header
#                  edit recompiles that many plus main.cpp.
#   ON             each probe is its own object (slower, but a failure in one
#                  probe cannot cascade into a bucket), and the dispatcher is
#                  still one binary.

set(PROBE_SOURCES
    unary_probe.cpp
    binary_probe.cpp
    cast_probe.cpp
    qdq_probe.cpp
    softmax_probe.cpp
    rmsnorm_probe.cpp
    layernorm_probe.cpp
    prelu_probe.cpp
    resize_probe.cpp
    expand_probe.cpp
    gap_probe.cpp
    rotary_probe.cpp
    embedding_probe.cpp
    reduce_transpose_probe.cpp
    downact_probe.cpp
    int64_probe.cpp
    dead_shader_probe.cpp
    linear_attn_probe.cpp
    matmul_probe.cpp
    matmul_i8_probe.cpp
    matmul_ada_probe.cpp
    matmul_route_probe.cpp
    descriptor_buffer_probe.cpp
    # Migrated from tests/ once the dispatcher grew an argv passthrough: these
    # three take parameters (a .vkopbin, n/k/reps, full|tiny) rather than
    # running a fixed sweep.
    image_layout_probe.cpp
    min_max_probe.cpp
    readback_probe.cpp
)

# A bucket count of 1 would be fastest but puts all 26 in one TU. Four keeps
# the per-header-edit cost near the measured 4-probe merge (10.2s) while
# limiting how much a single broken probe takes down with it.
set(PROBE_BUCKETS 4 CACHE STRING
    "Number of translation units the probe sources are merged into")

set(PROBE_TU_SOURCES)
if(PROBE_BUCKETS GREATER 1)
    list(LENGTH PROBE_SOURCES _n)
    math(EXPR _last "${PROBE_BUCKETS} - 1")
    foreach(_b RANGE ${_last})
        set(_tu ${CMAKE_CURRENT_BINARY_DIR}/probes_tu${_b}.cpp)
        file(WRITE ${_tu}
            "// Auto-generated - do not edit.\n"
            "// Round-robin merge of the probe sources; see CMakeLists.txt.\n")
        set(_i 0)
        foreach(_src ${PROBE_SOURCES})
            math(EXPR _mod "${_i} % ${PROBE_BUCKETS}")
            if(_mod EQUAL ${_b})
                file(APPEND ${_tu}
                    "#include \"${CMAKE_CURRENT_SOURCE_DIR}/${_src}\"\n")
            endif()
            math(EXPR _i "${_i} + 1")
        endforeach()
        list(APPEND PROBE_TU_SOURCES ${_tu})
    endforeach()
else()
    set(PROBE_TU_SOURCES ${PROBE_SOURCES})
endif()

add_executable(vkop_probes main.cpp ${PROBE_TU_SOURCES})
set_source_files_properties(${PROBE_TU_SOURCES} PROPERTIES GENERATED TRUE)
target_include_directories(vkop_probes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vkop_probes PRIVATE vkop vload)
target_compile_features(vkop_probes PRIVATE cxx_std_17)

# Match the rest of the tree: no optimisation, and -g1 rather than -g (the
# probes are debugged by running, not by single-stepping locals; -g1 keeps
# every backtrace line-accurate at half the object size).
if(MSVC)
    target_compile_options(vkop_probes PRIVATE /wd4100)
else()
    target_compile_options(vkop_probes PRIVATE
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-function
        -g1
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(vkop_probes PRIVATE -Wno-dangling-reference)
    endif()
endif()

# The probes build their ops directly, so they need the build tree's shader
# archive, exactly as the old per-probe targets did.
target_compile_definitions(vkop_probes PRIVATE
    VKOP_DEV_SHADER_ARCHIVE="${VKOP_SHADER_ARCHIVE_FILE}")
add_dependencies(vkop_probes shader_archive)

# Probes are diagnostics, not a ctest suite (see tests/CMakeLists.txt for the
# reason). Build them on demand or through the `probes` target; they are not
# part of `all`, so a header edit does not rebuild them.
install(TARGETS vkop_probes RUNTIME DESTINATION bin OPTIONAL)
