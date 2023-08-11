cmake_minimum_required (VERSION 3.17 FATAL_ERROR)

#---[ CMake Config ]--------------------
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

message(STATUS "Using CMake version ${CMAKE_VERSION}")
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0056 NEW)
#=======================================

#---[ Build Config ]--------------------
project(OCCA
  VERSION 1.6.0
  DESCRIPTION  "JIT Compilation for Multiple Architectures: C++, OpenMP, CUDA, HIP, OpenCL, Metal"
  HOMEPAGE_URL "https://github.com/libocca/occa"
  LANGUAGES    C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# CMake will decay to a previous C++ standard if a compiler does not support C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_OPENMP "Build with OpenMP if available" ON)
option(ENABLE_CUDA   "Build with CUDA if available"   ON)
option(ENABLE_OPENCL "Build with OpenCL if available" ON)
option(ENABLE_HIP    "Build with HIP if available" ON)
option(ENABLE_METAL  "Build with Metal if available" ON)
option(ENABLE_DPCPP "Build with SYCL/DPCPP if available" ON)

option(ENABLE_TESTS    "Build tests"               OFF)
option(ENABLE_EXAMPLES "Build simple examples"     OFF)
option(ENABLE_FORTRAN  "Enable Fortran interface"  OFF)

if(ENABLE_FORTRAN)
  enable_language(Fortran)
endif()

option(ENABLE_SHARABLE_DEVICE  "Enable sharable device by multiple threads"  OFF)
if (ENABLE_SHARABLE_DEVICE)
  add_compile_definitions(OCCA_THREAD_SHARABLE_ENABLED=1)
  message("-- OCCA sharable by multi-threads     : Enabled")
else()
  add_compile_definitions(OCCA_THREAD_SHARABLE_ENABLED=0)
endif()

set(OCCA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OCCA_BUILD_DIR  ${CMAKE_BINARY_DIR})

set(OCCA_USING_VS OFF)
set(OCCA_UNSAFE   OFF)

# Test Apple first because UNIX==true for Apple and Linux.
if(APPLE)
  set(OCCA_OS "OCCA_MACOS_OS")
elseif(UNIX)
  set(OCCA_OS "OCCA_LINUX_OS")
else()
  set(OCCA_OS "OCCA_WINDOWS_OS")
endif()

include(SetCompilerFlags)
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-fno-strict-aliasing" COMPILER_SUPPORTS_NO_STRICT_ALIASING)
if(COMPILER_SUPPORTS_NO_STRICT_ALIASING)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
endif()

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
message("-- System     : ${CMAKE_SYSTEM}")
message("-- Build type : ${CMAKE_BUILD_TYPE}")
message("-- C flags    : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
message("-- CXX flags  : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
if(ENABLE_FORTRAN)
  message("-- F90 flags  : ${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
endif()

if(ENABLE_FORTRAN)
  set(CMAKE_Fortran_MODULE_DIRECTORY ${OCCA_BUILD_DIR}/mod)
endif()
#=======================================

#---[ libocca.so ]----------------------
add_library(libocca SHARED)

# Without this, CMake will create liblibocca.so
set_target_properties(libocca PROPERTIES
  OUTPUT_NAME occa
  LIBRARY_OUTPUT_DIRECTORY ${OCCA_BUILD_DIR}/lib)

# Find needed and requested packages
find_package(Threads REQUIRED)

# Use the provided imported target Threads::Threads, to make our package relocatable
target_link_libraries(libocca PRIVATE
  Threads::Threads ${CMAKE_DL_LIBS})

target_include_directories(libocca PUBLIC
  $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${OCCA_BUILD_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_include_directories(libocca PRIVATE
  $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)

target_compile_definitions(libocca PRIVATE -DUSE_CMAKE)
#=======================================

#---[ OpenMP ]--------------------------
if(ENABLE_OPENMP)
  find_package(OpenMP)

  if(OPENMP_CXX_FOUND)
    set(OCCA_OPENMP_ENABLED 1)

    message("-- OpenMP include dirs: ${OpenMP_CXX_INCLUDE_DIRS}")
    message("-- OpenMP libraries:    ${OpenMP_CXX_LIBRARIES}")

    # Use the provided imported target OpenMP::OpenMP_CXX,
    # (which wraps the CXX_INCLUDE_DIRS and CXX_LIBRARIES,) to make our package relocatable
    target_link_libraries(libocca PRIVATE OpenMP::OpenMP_CXX)
  else(OPENMP_CXX_FOUND)
    set(OCCA_OPENMP_ENABLED 0)
  endif(OPENMP_CXX_FOUND)
endif(ENABLE_OPENMP)
#=======================================

#---[ CUDA ]----------------------------
if(ENABLE_CUDA)
  find_package(CUDAToolkit)

  if(CUDAToolkit_FOUND)
    set(OCCA_CUDA_ENABLED 1)

    message("-- CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    message("-- CUDA driver library: ${CUDAToolkit_LIBRARY_DIR}")

    # Use the provided imported target CUDA::cuda_driver, to make our package relocatable
    target_link_libraries(libocca PRIVATE CUDA::cuda_driver)

  else (CUDAToolkit_FOUND)
    set(OCCA_CUDA_ENABLED 0)
  endif(CUDAToolkit_FOUND)
endif(ENABLE_CUDA)
#=======================================

#---[ OpenCL ]--------------------------
if(ENABLE_OPENCL)
  find_package(OpenCLWrapper)

  if(OpenCL_FOUND)
    set(OCCA_OPENCL_ENABLED 1)

    message("-- OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
    message("-- OpenCL libraries:    ${OpenCL_LIBRARIES}")

    # Use the provided imported target OpenCL::OpenCL,
    # (which wraps the _INCLUDE_DIRS and _LIBRARIES,) to make our package relocatable
    target_link_libraries(libocca PRIVATE OpenCL::OpenCL)
  else (OpenCL_FOUND)
    set(OCCA_OPENCL_ENABLED 0)
  endif(OpenCL_FOUND)
endif(ENABLE_OPENCL)
#=======================================

#---[ SYCL/DPCPP ]-----------------------
if(ENABLE_DPCPP)
  find_package(DPCPP)

  if(DPCPP_FOUND)
    set(OCCA_DPCPP_ENABLED 1)

    message("-- DPCPP flags: ${SYCL_FLAGS}")
    message("-- DPCPP include dirs: ${SYCL_INCLUDE_DIRS}")
    message("-- DPCPP libraries:    ${SYCL_LIBRARIES}")

    # Use our wrapper imported target OCCA::depends::DPCPP,
    # (which wraps the _INCLUDE_DIRS and _LIBRARIES,) to make our package relocatable
    target_link_libraries(libocca PRIVATE OCCA::depends::DPCPP)
  else()
    set(OCCA_DPCPP_ENABLED 0)
  endif()
endif(ENABLE_DPCPP)
#=======================================

#---[ HIP ]-----------------------------
if(ENABLE_HIP)
  find_package(HIP)

  if(HIP_FOUND)
    set(OCCA_HIP_ENABLED 1)

    message("-- HIP version:      ${HIP_VERSION_STRING}")
    message("-- HIP platform:     ${HIP_PLATFORM}")
    message("-- HIP include dirs: ${HIP_INCLUDE_DIRS}")
    message("-- HIP libraries:    ${HIP_LIBRARIES}")

    # Use our wrapper imported target OCCA::depends::HIP,
    # (which wraps the _COMPILE_DEFINITIONS, _INCLUDE_DIRS and _LIBRARIES,) to make our package relocatable
    target_link_libraries(libocca PRIVATE OCCA::depends::HIP)
  else (HIP_FOUND)
    set(OCCA_HIP_ENABLED 0)
  endif(HIP_FOUND)
endif(ENABLE_HIP)
#=======================================

#---[ Metal ]---------------------------
if(ENABLE_METAL AND APPLE)
  find_package(METAL)

  if(METAL_FOUND)
    set(OCCA_METAL_ENABLED 1)

    message("-- METAL libraries:     ${METAL_LIBRARY}")
    message("-- METAL core services: ${CORE_SERVICES}")
    message("-- METAL app kit:       ${APP_KIT}")

    # Use our wrapper imported target OCCA::depends::METAL,
    # (which wraps the three libraries variables,) to make our package relocatable
    target_link_libraries(libocca PRIVATE OCCA::depends::METAL)
  else()
    set(OCCA_METAL_ENABLED 0)
  endif()
endif()
#=======================================

# Generate CompiledDefines from libraries we found
configure_file(
  scripts/build/compiledDefinesTemplate.hpp.in
  ${OCCA_BUILD_DIR}/include/occa/defines/compiledDefines.hpp)

install(
  FILES ${OCCA_BUILD_DIR}/include/occa/defines/compiledDefines.hpp
  DESTINATION include/occa/defines)

# Find source files
file(
  GLOB_RECURSE OCCA_SRC_cpp
  RELATIVE ${OCCA_SOURCE_DIR} "src/*.cpp")

if(ENABLE_FORTRAN)
  file(GLOB_RECURSE OCCA_SRC_f90
    RELATIVE ${OCCA_SOURCE_DIR} "src/*.f90")

  install(CODE
    "file(GLOB public-modules ${CMAKE_Fortran_MODULE_DIRECTORY}/*.mod)\n
     file(INSTALL DESTINATION
       ${CMAKE_INSTALL_PREFIX}/include
       TYPE FILE
       FILES \${public-modules})")

  set_target_properties(libocca PROPERTIES
    LINKER_LANGUAGE C)
endif()

if(OCCA_METAL_ENABLED)
  file(GLOB_RECURSE OCCA_SRC_metal
    RELATIVE ${OCCA_SOURCE_DIR} "src/*.mm")
endif()

set(OCCA_SRC
  ${OCCA_SRC_cpp}
  ${OCCA_SRC_f90}
  ${OCCA_SRC_metal})

target_sources(libocca PRIVATE ${OCCA_SRC})

install(TARGETS libocca EXPORT occaExport DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

if(ENABLE_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif(ENABLE_TESTS)

if(ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif(ENABLE_EXAMPLES)

add_subdirectory(bin)

# Create a package config and associated files.
include(ExportAndPackageConfig)

install(CODE
  "configure_file(
    ${OCCA_SOURCE_DIR}/modulefiles/occa
    ${CMAKE_INSTALL_PREFIX}/modulefiles/occa
    @ONLY
  )"
)
