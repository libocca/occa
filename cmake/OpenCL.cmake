find_package(OpenCL)

if (OpenCL_FOUND)
  set(WITH_OPENCL 1)
  set(OCCA_OPENCL_ENABLED 1)
  include_directories( ${OpenCL_INCLUDE_DIRS} )
endif()
