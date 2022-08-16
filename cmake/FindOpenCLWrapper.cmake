###############################################################################
# FIND module wrapper around finding OpenCL
# This Find module is also distributed alongside the occa package config file!
###############################################################################

# Try finding OpenCL. The user should set OpenCL_ROOT if needed.
find_package(OpenCL QUIET)
if(NOT OpenCL_FOUND)
  # Otherwise, look for the headers and library in standard locations
  find_path(OpenCL_INCLUDE_DIR
    NAMES CL/cl.h OpenCL/cl.h
    PATHS
      ENV CUDA_PATH
      ENV CUDAToolkit_ROOT
      ENV ROCM_PATH
      ENV NVHPC_ROOT
      ENV SYCL_ROOT
      /usr/local/cuda
      /opt/rocm/opencl
      /opt/intel/oneapi/compiler/latest/linux
    PATH_SUFFIXES
      include
      include/sycl
  )

  find_library(OpenCL_LIBRARY
    NAMES OpenCL libOpenCL
    PATHS
      ENV CUDA_PATH
      ENV CUDAToolkit_ROOT
      ENV ROCM_PATH
      ENV NVHPC_ROOT
      ENV SYCL_ROOT
      /usr/local/cuda
      /opt/rocm/opencl
      /opt/intel/oneapi/compiler/latest/linux
    PATH_SUFFIXES
      lib
      lib64
  )
endif()
find_package(OpenCL)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    OpenCLWrapper
    REQUIRED_VARS
    OpenCL_FOUND
    )
