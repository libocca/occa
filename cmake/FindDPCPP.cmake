###############################################################################
# FIND module for DPCPP components
# This Find module is also distributed alongside the occa package config file!
###############################################################################

message(CHECK_START "Looking for DPC++")
unset(missingDpcppComponents)

find_path(
  SYCL_INCLUDE_DIRS
  NAMES
    CL/sycl.hpp
  PATHS
    /opt/intel/oneapi/compiler/latest/linux
    ENV SYCL_ROOT
    ${SYCL_ROOT}
  PATH_SUFFIXES
    include/sycl
)

find_path(
  SYCL_EXT_DIRS
  NAMES
    sycl/ext
  PATHS
    /opt/intel/oneapi/compiler/latest/linux
    ENV SYCL_ROOT
    ${SYCL_ROOT}
  PATH_SUFFIXES
    include
)

find_library(
  SYCL_LIBRARIES
  NAMES
    sycl libsycl
  PATHS
    /opt/intel/oneapi/compiler/latest/linux
    ENV SYCL_ROOT
    ${SYCL_ROOT}
  PATH_SUFFIXES
    lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    DPCPP
    REQUIRED_VARS
    SYCL_INCLUDE_DIRS
    SYCL_EXT_DIRS
    SYCL_LIBRARIES
    )

if(DPCPP_FOUND AND NOT TARGET OCCA::depends::DPCPP)
  # Create our wrapper imported target
  # Put it in the OCCA namespace to make it clear that we created it.
  add_library(OCCA::depends::DPCPP INTERFACE IMPORTED)
  set_target_properties(OCCA::depends::DPCPP PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SYCL_INCLUDE_DIRS};${SYCL_EXT_DIRS}"
    INTERFACE_LINK_LIBRARIES "${SYCL_LIBRARIES}"
  )
endif()
