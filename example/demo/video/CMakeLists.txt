option(USE_GLFW "Use GLFW for window creation" ON)
option(GLFW_USE_SYSTEM "Use system installed GLFW" ON)

find_package(Vulkan REQUIRED)

if(USE_GLFW)
    if(GLFW_USE_SYSTEM)
        find_package(glfw3 REQUIRED)
        message(STATUS "Using system GLFW")
    else()
        include(FetchContent)
        
        FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.3.8
            GIT_SHALLOW TRUE
        )
        
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

        if(ANDROID)
            set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
            set(GLFW_BUILD_X11 OFF CACHE BOOL "" FORCE)
        endif()
        
        FetchContent_MakeAvailable(glfw)
        message(STATUS "Using downloaded GLFW")
    endif()
endif()



add_executable(vkdeo main.cpp)
target_link_libraries(vkdeo PRIVATE vkop)

if(USE_GLFW)
    target_link_libraries(vkdeo PUBLIC glfw)
    target_compile_definitions(vkdeo PUBLIC USE_GLFW)
endif()

if(ANDROID)
    target_link_libraries(vkdeo PRIVATE android log)
endif()