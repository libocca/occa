###############################################################################
# FIND module wrapper around finding OpenCL
# This Find module is also distributed alongside the occa package config file!
###############################################################################

# Look in some default places for OpenCL and set OpenCL_ROOT if not already set
if(NOT OpenCL_ROOT)
  # Search in user specified path first
  find_path(OpenCL_ROOT
    NAMES CL/cl.h
    PATHS
    ENV   OpenCL_PATH
    DOC   "OpenCL root location"
    NO_DEFAULT_PATH)

  # Now search in default path
  find_path(OpenCL_ROOT
    NAMES CL/cl.h
    PATHS 
      /usr 
      /opt/rocm/opencl 
      /usr/local/cuda 
      /opt/intel/oneapi/compiler/latest/linux
    PATH_SUFFIXES sycl
    DOC   "OpenCL root location")
endif()

# Trick CMake's default OpenCL module to look in our directory
set(ENV{AMDAPPSDKROOT} ${OpenCL_ROOT})

find_package(OpenCL)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    OpenCLWrapper
    REQUIRED_VARS
    OpenCL_FOUND
    )
