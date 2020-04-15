#Look in some default places for OpenCL and set OPENCL_ROOT

# Search in user specified path first
find_path(
  OPENCL_ROOT
  NAMES include/CL/cl.h
  PATHS
  ENV OPENCL_PATH
  DOC "OPENCL root location"
  NO_DEFAULT_PATH
  )
# Now search in default path
find_path(
  OPENCL_ROOT
  NAMES include/CL/cl.h
  PATHS
  /usr/
  /opt/rocm/opencl
  /usr/local/cuda
  DOC "OPENCL root location"
  )

#Trick cmake's default OpenCL module to look in our directory
set(ENV{AMDAPPSDKROOT} ${OPENCL_ROOT})

find_package(OpenCL)

if (OpenCL_FOUND)
  set(WITH_OPENCL 1)
  set(OCCA_OPENCL_ENABLED 1)
  include_directories( ${OpenCL_INCLUDE_DIRS} )
endif()
