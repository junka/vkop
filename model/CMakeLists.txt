cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FlatBuffers schema → C++ header codegen.
# The generated header (model/generated/vkop_model_generated.h) and the vendored
# flatbuffers runtime headers (model/generated/flatbuffers/*.h) are committed, so
# building does NOT require flatc to be installed. When flatc IS available, the
# custom target below regenerates the header from model/vkop_model.fbs.
find_program(FLATC flatc)

set(FBS_SCHEMA ${CMAKE_CURRENT_SOURCE_DIR}/vkop_model.fbs)
set(FBS_GENERATED_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/generated/vkop_model_generated.h)

if(FLATC)
    add_custom_command(
        OUTPUT ${FBS_GENERATED_HEADER}
        COMMAND ${FLATC} --cpp --gen-mutable --gen-object-api --reflect-names
                        -o ${CMAKE_CURRENT_SOURCE_DIR}/generated
                        ${FBS_SCHEMA}
        DEPENDS ${FBS_SCHEMA}
        COMMENT "Regenerating FlatBuffers C++ header from ${FBS_SCHEMA}"
        VERBATIM
    )
    add_custom_target(generate_flatbuffers DEPENDS ${FBS_GENERATED_HEADER})
endif()

#add_executable(vload load.cpp)
add_library(vload load.cpp)

# generated/ holds both the schema-generated header and the vendored flatbuffers
# runtime headers. Mark SYSTEM so -Wall/-Wextra/-Werror do not fire on third-party
# (generated + vendored) code.
target_include_directories(vload
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/generated
)

if(FLATC)
    add_dependencies(vload generate_flatbuffers)
endif()
