find_package(HIP)

if (HIP_FOUND)
  set(WITH_HIP 1)
  set(OCCA_HIP_ENABLED 1)
  add_definitions(-D${HIP_RUNTIME_DEFINE})
  include_directories( ${HIP_INCLUDE_DIRS} )

else (HIP_FOUND)
  set(WITH_HIP 0)
  set(OCCA_HIP_ENABLED 0)
endif(HIP_FOUND)
