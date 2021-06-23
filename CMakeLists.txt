cmake_minimum_required(VERSION 3.20)

project(OpenMP_Offloading_to_SYCL LANGUAGES CXX)


message(STATUS "Using \"${SYCL_IMPLEMENTATION}\" as SYCL implementation.")


# get target hardware
# omp = CPU, cuda = NVIDIA GPUs, hip = AMD GPUs, spirv = Intel GPUs
if (DEFINED SYCL_TARGET)
  set(SYCL_TARGET_LIST ${SYCL_TARGET})
elseif (DEFINED ENV{SYCL_TARGET})
  set(SYCL_TARGET_LIST $ENV{SYCL_TARGET})
else ()
  message(FATAL_ERROR "Either the command line argument '-DSYCL_TARGET' must be present or the environment variable 'SYCL_TARGET' must be defined!")
endif ()
list(LENGTH SYCL_TARGET_LIST SYCL_TARGET_LIST_LEN)
if (NOT ${SYCL_TARGET_LIST_LEN} MATCHES 1)
  message(FATAL_ERROR "Currently only exactly one target architecture is supported!")
endif ()
string(FIND "${SYCL_TARGET_LIST}" ":" SYCL_ARCH_POS)
string(SUBSTRING "${SYCL_TARGET_LIST}" 0 ${SYCL_ARCH_POS} SYCL_ARCH)
set(SYCL_TARGETS omp cuda hip spirv)
list(FIND SYCL_TARGETS "${SYCL_ARCH}" SYCL_TARGET_IDX)
if (${SYCL_TARGET_IDX} MATCHES -1)
  message(FATAL_ERROR "Unrecognized target \"${SYCL_ARCH}\"!")
endif ()
message(STATUS "Using \"${SYCL_ARCH}\" as SYCL target architecture.")


# find different SYCL implementations
if (SYCL_IMPLEMENTATION MATCHES "hipsycl")

  set(HIPSYCL_TARGETS ${SYCL_TARGET_LIST})

  find_package(hipSYCL CONFIG REQUIRED)
  message(STATUS "Found hipSYCL.")

elseif (SYCL_IMPLEMENTATION MATCHES "dpcpp")

  message(STATUS "Found dpcpp.")

endif ()


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



# set SYCL implementation specific options
if (SYCL_IMPLEMENTATION MATCHES "hipsycl")

  add_sycl_to_target(TARGET ${EXECUTABLE_NAME} SOURCES hodgkin_huxley_1952_gpu_fast_monodomain.0.cpp)

elseif (SYCL_IMPLEMENTATION MATCHES "dpcpp")

  target_compile_options(${EXECUTABLE_NAME} PRIVATE -sycl-std=2020)
  target_compile_options(${EXECUTABLE_NAME} PRIVATE -fsycl)
  target_link_libraries(${EXECUTABLE_NAME} PRIVATE -fsycl)

  if (SYCL_ARCH MATCHES "omp")
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice)
  elseif (SYCL_ARCH MATCHES "cuda")
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
  endif ()

endif ()




# CMake options
option(DEBUG_INFO "Print debugging information." OFF)
if (DEBUG_INFO)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE DEBUG_INFO)
endif ()

option(ENABLE_TIMING "Enable timing for the two important functions." OFF)
if (ENABLE_TIMING)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE ENABLE_TIMING)
endif ()

