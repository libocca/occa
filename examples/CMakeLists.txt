cmake_minimum_required (VERSION 3.21 FATAL_ERROR)
project(OCCAExamples
  VERSION 2.0.0
  DESCRIPTION  "OCCA Examples"
  HOMEPAGE_URL "https://github.com/libocca/occa"
  LANGUAGES C CXX)

if (NOT TARGET OCCA::libocca)
  find_package(OCCA REQUIRED)
endif()

macro(add_test_with_mode exe mode device)
  add_test(NAME ${exe}-${mode} COMMAND ./${exe} --verbose --device "${device}")
  set_property(TEST ${exe}-${mode} APPEND PROPERTY ENVIRONMENT OCCA_CACHE_DIR=${OCCA_BUILD_DIR}/occa)
endmacro()

macro(add_test_with_modes exe)
  add_test_with_mode(${exe} serial "{\"mode\": \"Serial\"}")
  if (OCCA_CUDA_ENABLED)
    add_test_with_mode(${exe} cuda "{\"mode\": \"CUDA\", \"device_id\": 0}")
  endif()
  if (OCCA_HIP_ENABLED)
    add_test_with_mode(${exe} hip "{\"mode\": \"HIP\", \"device_id\": 0}")
  endif()
  if (OCCA_METAL_ENABLED)
    add_test_with_mode(${exe} metal "{\"mode\": \"Metal\", \"device_id\": 0}")
  endif()
  if (OCCA_OPENCL_ENABLED)
    add_test_with_mode(${exe} opencl "{\"mode\": \"OpenCL\", \"platform_id\": 0, \"device_id\": 0}")
  endif()
  if (OCCA_DPCPP_ENABLED)
    add_test_with_mode(${exe} dpcpp "{\"mode\": \"dpcpp\", \"platform_id\": 0, \"device_id\": 0}")
  endif()
  if (OCCA_OPENMP_ENABLED)
    add_test_with_mode(${exe} openmp "{\"mode\": \"OpenMP\"}")
  endif()
endmacro()

macro(add_test_without_mode exe)
  add_test(NAME ${exe} COMMAND ${exe} --verbose)
  set_property(TEST ${exe} APPEND PROPERTY ENVIRONMENT OCCA_CACHE_DIR=${OCCA_BUILD_DIR}/occa)
endmacro()

macro(compile_example target file mode)
  add_executable(${target} ${file})
  target_link_libraries(${target} OCCA::libocca)
  target_include_directories(${target} PRIVATE $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)

  file(RELATIVE_PATH install_dir ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR})
  file(GLOB_RECURSE OKLS "${CMAKE_CURRENT_LIST_DIR}/*.okl")
  install(TARGETS ${target} DESTINATION ${install_dir})
  install(FILES ${OKLS} DESTINATION ${install_dir})

  if (NOT DEFINED OCCA_ENABLE_TESTS)
    return()
  endif()
  if (${mode})
    add_test_with_modes(${target})
  else()
    add_test_without_mode(${target})
  endif()
endmacro()

macro(compile_c_example target file)
  compile_example(examples_c_${target} ${file} TRUE)
endmacro()

macro(compile_cpp_example target file)
  compile_example(examples_cpp_${target} ${file} FALSE)
endmacro()

macro(compile_cpp_example_with_modes target file)
  compile_example(examples_cpp_${target} ${file} TRUE)
endmacro()

macro(compile_fortran_example_with_modes target file)
  compile_example(examples_fortran_${target} ${file} TRUE)
endmacro()

add_subdirectory(c)
add_subdirectory(cpp)
if (OCCA_ENABLE_FORTRAN)
  add_subdirectory(fortran)
endif()
