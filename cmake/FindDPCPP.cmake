###############################################################################
# FIND module for DPCPP components
# This Find module is also distributed alongside the occa package config file!
###############################################################################

message(CHECK_START "Looking for DPC++")
unset(missingDpcppComponents)

cmake_path(CONVERT "${CMAKE_CXX_COMPILER}" TO_CMAKE_PATH_LIST compiler_path)
cmake_path(GET compiler_path PARENT_PATH compiler_bin_dir)
cmake_path(GET compiler_bin_dir PARENT_PATH compiler_root_dir)

find_path(
  SYCL_INCLUDE_DIRS
  NAMES
    sycl/sycl.hpp
  PATHS
    ENV SYCL_ROOT
    ${SYCL_ROOT}
    ${compiler_root_dir}
    ENV CMPLR_ROOT
  PATH_SUFFIXES
    include
    include/sycl
    include/CL
    include/sycl/CL
)

find_library(
  SYCL_LIBRARIES
  NAMES
    sycl libsycl
  PATHS
    ENV SYCL_ROOT
    ${SYCL_ROOT}
    ${compiler_root_dir}
    ENV CMPLR_ROOT
  PATH_SUFFIXES
    lib
)

if(NOT OCCA_DPCPP_COMPILER_FLAGS)
  if(DEFINED ENV{OCCA_DPCPP_COMPILER_FLAGS})
    set(SYCL_FLAGS $ENV{OCCA_DPCPP_COMPILER_FLAGS})
  else()
    set(SYCL_FLAGS -fsycl)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    DPCPP
    REQUIRED_VARS
    SYCL_INCLUDE_DIRS
    SYCL_LIBRARIES
    SYCL_FLAGS
    )

if(DPCPP_FOUND AND NOT TARGET OCCA::depends::DPCPP)
  # Create our wrapper imported target
  # Put it in the OCCA namespace to make it clear that we created it.
  add_library(OCCA::depends::DPCPP INTERFACE IMPORTED)
  separate_arguments(SYCL_FLAGS UNIX_COMMAND "${SYCL_FLAGS}")
  target_compile_options(OCCA::depends::DPCPP INTERFACE ${SYCL_FLAGS})
  set_target_properties(OCCA::depends::DPCPP PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SYCL_INCLUDE_DIRS};${SYCL_INCLUDE_DIRS}/sycl"
    INTERFACE_LINK_LIBRARIES "${SYCL_LIBRARIES}"
  )
endif()
