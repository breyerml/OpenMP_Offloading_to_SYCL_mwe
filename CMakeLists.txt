cmake_minimum_required(VERSION 3.16.3)

project(OpenMP_Offloading_to_SYCL LANGUAGES CXX)


set(NVIDIA_GPU "" CACHE STRING "")
set(AMD_GPU "" CACHE STRING "")
if(NOT NVIDIA_GPU STREQUAL "")
    set(HIPSYCL_TARGETS "cuda:${NVIDIA_GPU}")
elseif (NOT AMD_GPU STREQUAL "")
    set(HIPSYCL_TARGETS "hip:${AMD_GPU}")
else()
    set(HIPSYCL_TARGETS "omp")
endif()

# check if hipSYCL is used as SYCL compiler
find_package(hipSYCL CONFIG)
if(hipSYCL_FOUND)
    set(SYCL_COMPILER "hipSYCL" CACHE STRING "" FORCE)
    message(STATUS "found hipSYCL")
else()
    # if not, check if DPC++ is used instead
    try_compile(SYCL_CHECK_FOR_DPCPP_COMPILER
            "${CMAKE_BINARY_DIR}/temp"
            "${PROJECT_SOURCE_DIR}/cmake/test_sycl_dpcpp.cpp"
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON)
    if(SYCL_CHECK_FOR_DPCPP_COMPILER)
        set(SYCL_COMPILER "DPC++" CACHE STRING "" FORCE)
        message(STATUS "found DPC++")
    else()
        message(FATAL_ERROR "Cannot find a suitable SYCL compiler!")
    endif()
endif()


# set source files
set(SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
        hodgkin_huxley_1952_gpu_fast_monodomain.0.cpp)


# create executable
set(EXECUTABLE_NAME prog)
add_executable(${EXECUTABLE_NAME} ${SOURCE})


# set include directory
target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

# set important compiler flags
target_compile_features(${EXECUTABLE_NAME} PRIVATE cxx_std_17)
target_link_options(${EXECUTABLE_NAME} PRIVATE -lpthread) # maybe unnecessary

# enable warnings
target_compile_options(${EXECUTABLE_NAME} PRIVATE -Wall -Wextra)
# enable non IEEE conformant floating point arithmetic
#target_compile_options(${EXECUTABLE_NAME} PRIVATE -ffast-math)



# link against OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_link_options(${EXECUTABLE_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif ()



# set additional properties depending on the used SYCL implementation
if("${SYCL_COMPILER}" STREQUAL "hipSYCL")
    add_sycl_to_target(TARGET ${EXECUTABLE_NAME} SOURCES hodgkin_huxley_1952_gpu_fast_monodomain.0.cpp)
    # silence unknown options warnings
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -Wno-unknown-warning-option)
    message(STATUS "HIPSYCL_TARGETS: \"${HIPSYCL_TARGETS}\"")
elseif("${SYCL_COMPILER}" STREQUAL "DPC++")
    # enable DPC++ SYCL support
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -sycl-std=2020)
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -fsycl)
    target_link_options(${EXECUTABLE_NAME} PRIVATE -fsycl)


    # nvidia targets
    if(NOT NVIDIA_GPU STREQUAL "")
        target_compile_options(${EXECUTABLE_NAME} PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
        target_link_options(${EXECUTABLE_NAME} PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
    endif()
    # amd targets
    if(NOT AMD_GPU STREQUAL "")
        target_compile_options(${EXECUTABLE_NAME} PRIVATE -fsycl-targets=amdgcn-amd-amdhsa)
        target_link_options(${EXECUTABLE_NAME} PRIVATE -fsycl-targets=amdgcn-amd-amdhsa)
        target_compile_options(${EXECUTABLE_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${AMD_GPU})
        target_link_options(${EXECUTABLE_NAME} PRIVATE -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${AMD_GPU})
    endif()
endif()



# CMake options
option(DEBUG_INFO "Print debugging information." OFF)
if (DEBUG_INFO)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE DEBUG_INFO)
endif ()

option(ENABLE_TIMING "Enable timing for the two important functions." OFF)
if (ENABLE_TIMING)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE ENABLE_TIMING)
endif ()

