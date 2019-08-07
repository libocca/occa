cmake_minimum_required (VERSION 3.0 FATAL_ERROR)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

message(STATUS "Using CMake version ${CMAKE_VERSION}")
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0056 NEW)

OPTION(USE_INTEL "Use Intel compilers" OFF)

IF (USE_INTEL)
    message("Using Intel compilers")
    include(IntelCompilers)
ENDIF()

# Set project
project (occa)

OPTION(ALLOW_CUDA "Allow CUDA" ON)
OPTION(ALLOW_MPI  "Allow MPI" OFF)

set(PROJ_DIR ${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${PROJ_DIR}/lib)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
SET(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(OpenCL)

# searches for CUDA if allowed
IF (ALLOW_CUDA)
    include(CUDA)
ENDIF()

IF (OpenCL_FOUND)
  SET(WITH_OPENCL 1)
  include_directories( ${OpenCL_INCLUDE_DIRS} )
ENDIF()

# searches for MPI if allowed
IF (ALLOW_MPI)
    include(MPI)
ENDIF()

include_directories(${PROJ_DIR}/include)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-c++11-long-long")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-long-long")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

include_directories(${PROJECT_BINARY_DIR})
# TODO:
# CONFIGURE_FILE( scripts/compiledDefinesTemplate.hpp.in ${PROJECT_BINARY_DIR}/occa/defines/compiledDefines.hpp )
# will not work because build directory unknown to runtime compiler
CONFIGURE_FILE( scripts/compiledDefinesTemplate.hpp.in ${PROJECT_SOURCE_DIR}/include/occa/defines/compiledDefines.hpp )

add_subdirectory(src)
add_subdirectory(bin)
# add_subdirectory(tests)
add_subdirectory(examples)
