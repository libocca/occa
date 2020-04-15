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

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

if (ALLOW_OPENMP)
  include(OpenMP)
endif()

if (ALLOW_OPENCL)
  include(OpenCL)
endif()

# searches for CUDA if allowed
if (ALLOW_CUDA)
  include(CUDA)
endif()

if (ALLOW_HIP)
  include(HIP)
endif()

# searches for MPI if allowed
if (ALLOW_MPI)
  include(MPI)
endif()

if (ALLOW_METAL)
  include(Metal)
endif()


add_definitions(-DUSE_CMAKE)

include(SetCompilerFlags)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if (ALLOW_OPENMP)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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
