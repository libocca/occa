cmake_minimum_required (VERSION 3.12 FATAL_ERROR)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

message(STATUS "Using CMake version ${CMAKE_VERSION}")
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0056 NEW)

# Set project
project (occa
  DESCRIPTION "JIT Compilation for Multiple Architectures: C++, OpenMP, CUDA, HIP, OpenCL, Metal"
  HOMEPAGE_URL "https://github.com/libocca/occa/"
  LANGUAGES C CXX)

include(OS)

option(ALLOW_CUDA "Allow CUDA" OFF)
option(ALLOW_MPI  "Allow MPI" OFF)
option(ALLOW_OPENMP "Allow OpenMP" OFF)
option(ALLOW_OPENCL "Allow OpenCL" OFF)
option(ALLOW_METAL "Allow Metal" OFF)
option(ALLOW_HIP "Allow HIP" OFF)
#option(ALLOW_ONEAPI "Allow OneAPI" OFF)
option(ENABLE_TESTS "Enable tests" OFF)

find_package(Threads REQUIRED)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

if (ALLOW_OPENMP)
  include(OpenMP)
endif()

if (ALLOW_OPENCL)
  include(OpenCL)
endif()

# searches for CUDA if allowed
if (ALLOW_CUDA)
  set(OCCA_CUDA_ENABLED 1)
  set(WITH_CUDA 1)
  enable_language(CUDA)
endif()

# searches for MPI if allowed
if (ALLOW_MPI)
  include(MPI)
endif()

if (ALLOW_METAL)
  include(Metal)
endif()

if (ALLOW_HIP)
#    set(OCCA_HIP_ENABLED 1)
#    set(WITH_HIP 1)
#    include(ROCm) ... not available, TODO occa/cmake/FindROCm.cmake
endif()

add_definitions(-DUSE_CMAKE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-c++11-long-long")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-long-long")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

include_directories(${PROJECT_BINARY_DIR}/include)
set(OCCA_BUILD_DIR ${PROJECT_BINARY_DIR})
configure_file(scripts/compiledDefinesTemplate.hpp.in ${PROJECT_BINARY_DIR}/include/occa/defines/compiledDefines.hpp)
install(FILES ${PROJECT_BINARY_DIR}/include/occa/defines/compiledDefines.hpp DESTINATION include/occa/defines)

if (ENABLE_TESTS)
  include(CTest)
endif()

add_subdirectory(src)
add_subdirectory(bin)
add_subdirectory(examples)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

install(DIRECTORY include/ DESTINATION include)
