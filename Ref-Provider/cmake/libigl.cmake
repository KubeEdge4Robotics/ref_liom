if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.4.0
    # SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext
)
FetchContent_MakeAvailable(libigl)
