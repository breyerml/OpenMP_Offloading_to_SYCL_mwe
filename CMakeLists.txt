cmake_minimum_required(VERSION 3.13)

project(OpenMP_Offloading_to_SYCL)

# set source files
set(SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
        hodgkin_huxley_1952_gpu_fast_monodomain.0.cpp)

# create executable
add_executable(prog ${SOURCE})

# set include directory
target_include_directories(prog PUBLIC include)

# set important compiler flags
target_compile_features(prog PUBLIC cxx_std_17)
# TODO: uncomment for DPCPP SYCL2020 support
#target_compile_options(prog PUBLIC -sycl-std=2020)
target_compile_options(prog PRIVATE -Wall -Wextra)

# CMake options
option(DEBUG_INFO "Print debugging information." OFF)
if (DEBUG_INFO)
    target_compile_definitions(prog PUBLIC DEBUG_INFO)
endif ()

option(ENABLE_TIMING "Enable timing for the two important functions." OFF)
if (ENABLE_TIMING)
    target_compile_definitions(prog PUBLIC ENABLE_TIMING)
endif ()

# link against OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(prog PUBLIC OpenMP::OpenMP_CXX)
    target_link_options(prog PUBLIC ${OpenMP_CXX_FLAGS})
endif ()
