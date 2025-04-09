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

add_executable(main)
target_sources(main
    PRIVATE
    main.cpp
    PUBLIC FILE_SET HEADERS FILES
)

add_sycl_to_target(TARGET main)
target_compile_features(main
    PRIVATE
    cxx_std_26
    INTERFACE
    cxx_std_26
)
