###############################################################################
# FIND module for DPCPP components
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
