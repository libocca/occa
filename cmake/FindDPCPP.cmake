###############################################################################
# FIND module for DPCPP components
# This Find module is also distributed alongside the occa package config file!
###############################################################################

message(CHECK_START "Looking for DPC++")
unset(missingDpcppComponents)

find_package(IntelSYCL QUIET)
if (IntelSYCL_FOUND)
  set(DPCPP_FOUND TRUE)
  set(DPCPP_FLAGS "${SYCL_FLAGS}")
  set(DPCPP_INCLUDE_DIRS "${SYCL_INCLUDE_DIR};${SYCL_INCLUDE_SYCL_DIR}")
  set(DPCPP_LIBRARIES "${SYCL_LIBRARY}")
else()
  cmake_path(CONVERT "${CMAKE_CXX_COMPILER}" TO_CMAKE_PATH_LIST cxx_path)
  cmake_path(GET cxx_path PARENT_PATH cxx_bin_dir)
  cmake_path(GET cxx_bin_dir PARENT_PATH cxx_root_dir)

  find_path(
    DPCPP_INCLUDE_DIRS
    NAMES
      sycl/sycl.hpp
    PATHS
      ENV SYCL_ROOT
      ${SYCL_ROOT}
      ENV CMPLR_ROOT
      ${cxx_root_dir}
    PATH_SUFFIXES
      include
      include/sycl
      include/CL
      include/sycl/CL
  )
  set(DPCPP_INCLUDE_DIRS "${DPCPP_INCLUDE_DIRS};${DPCPP_INCLUDE_DIRS}/sycl")

  find_library(
    DPCPP_LIBRARIES
    NAMES
      sycl libsycl
    PATHS
      ENV SYCL_ROOT
      ${SYCL_ROOT}
      ENV CMPLR_ROOT
      ${cxx_root_dir}
    PATH_SUFFIXES
      lib
  )

  set(DPCPP_FLAGS -fsycl)
endif()

if (NOT SYCL_FLAGS)
  set(DPCPP_FLAGS SYCL_FLAGS)
endif()
# TODO: Check if the backend compiler supports DPCPP_FLAGS

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  DPCPP
  REQUIRED_VARS
  DPCPP_INCLUDE_DIRS
  DPCPP_LIBRARIES
  DPCPP_FLAGS
)

if (DPCPP_FOUND AND NOT TARGET OCCA::depends::DPCPP)
  # Create our wrapper imported target
  # Put it in the OCCA namespace to make it clear that we created it.
  add_library(OCCA::depends::DPCPP INTERFACE IMPORTED)
  separate_arguments(DPCPP_FLAGS UNIX_COMMAND "${DPCPP_FLAGS}")
  target_compile_options(OCCA::depends::DPCPP INTERFACE ${DPCPP_FLAGS})
  set_target_properties(OCCA::depends::DPCPP PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${DPCPP_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${DPCPP_LIBRARIES}"
  )
endif()
