# Require a recent version of cmake
cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

project(
    computational-physics-project
    VERSION 0.0.1
    DESCRIPTION "Final project of Aalto Computation Physics course PHYS-E0412 2025."
    LANGUAGES CXX
)

find_package(AdaptiveCpp REQUIRED)

# From AdpativeCpp repo examples folder:
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

If(NOT ACPP_DEBUG_LEVEL)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(ACPP_DEBUG_LEVEL 3 CACHE STRING "Choose the debug level, options are: \
0 (no debug), \
1 (print errors), \
2 (also print warnings), \
3 (also print general information)"
          FORCE)
    else()
        set(ACPP_DEBUG_LEVEL 2 CACHE STRING "Choose the debug level, options are: \
0 (no debug), \
1 (print errors), \
2 (also print warnings), \
3 (also print general information)"
          FORCE)
    endif()
endif()

# Main executable:

add_executable(main)
target_sources(main
    PRIVATE
    main.cpp
    w2_bssn_uniform_grid.cpp
    PRIVATE FILE_SET all_headers TYPE HEADERS FILES
    idg/einsum.hpp
    idg/generic_algorithm.hpp
    idg/sstd.hpp
    idg/tensor_network.hpp
    idg/string_manipulation.hpp
    tensor_buffer.hpp
    w2_bssn_uniform_grid.hpp
    ranges.hpp
    grid_types.hpp
    mdspan_utils.hpp
)

add_sycl_to_target(TARGET main)
target_compile_features(main
    PRIVATE
    cxx_std_26
)

include(FetchContent)
FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG stable
)
FetchContent_MakeAvailable(mdspan)
target_link_libraries(main PRIVATE std::mdspan)

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG v5.1.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glaze)

target_link_libraries(main PRIVATE glaze::glaze)


# Test executables:

add_executable(tensor_buffer)
target_sources(tensor_buffer
    PRIVATE
    test_tensor_buffer.cpp
    PRIVATE FILE_SET all_headers TYPE HEADERS
)

add_sycl_to_target(TARGET tensor_buffer)
target_compile_features(tensor_buffer
    PRIVATE
    cxx_std_26
)
target_link_libraries(tensor_buffer PRIVATE std::mdspan)
target_link_libraries(tensor_buffer PRIVATE glaze::glaze)


add_executable(w2_bssn_uniform_grid)
target_sources(w2_bssn_uniform_grid
    PRIVATE
    w2_bssn_uniform_grid.cpp
    w2_bssn_uniform_grid2.cpp
    test_w2_bssn_uniform_grid.cpp
    PRIVATE FILE_SET all_headers TYPE HEADERS
)

add_sycl_to_target(TARGET w2_bssn_uniform_grid)
target_compile_features(w2_bssn_uniform_grid
    PRIVATE
    cxx_std_26
)
target_link_libraries(w2_bssn_uniform_grid PRIVATE std::mdspan)
target_link_libraries(w2_bssn_uniform_grid PRIVATE glaze::glaze)


add_executable(mdspan_utils)
target_sources(mdspan_utils
    PRIVATE
    test_mdspan_utils.cpp
    PRIVATE FILE_SET all_headers TYPE HEADERS
)

add_sycl_to_target(TARGET mdspan_utils)
target_compile_features(mdspan_utils
    PRIVATE
    cxx_std_26
)
target_link_libraries(mdspan_utils PRIVATE std::mdspan)
target_link_libraries(mdspan_utils PRIVATE glaze::glaze)
