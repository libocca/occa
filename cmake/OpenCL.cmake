find_package(OpenCL)

IF (OpenCL_FOUND)
  SET(WITH_OPENCL 1)
  SET(OCCA_OPENCL_ENABLED 1)
  include_directories( ${OpenCL_INCLUDE_DIRS} )
ENDIF()
