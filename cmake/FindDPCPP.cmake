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
)

find_library(
  SYCL_LIBRARIES
  NAMES
    sycl libsycl
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    DPCPP
    REQUIRED_VARS
    SYCL_INCLUDE_DIRS
    SYCL_LIBRARIES
    )

if(DPCPP_FOUND and NOT TARGET OCCA::depends::DPCPP)
  # Create our wrapper imported target
  # Put it in the OCCA namespace to make it clear that we created it.
  add_library(OCCA::depends::DPCPP INTERFACE IMPORTED)
  set_target_properties(OCCA::depends::DPCPP PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SYCL_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${SYCL_LIBRARIES}"
  )
endif()
