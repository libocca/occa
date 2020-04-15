find_package(CUDA)

if (CUDA_FOUND)
  #find the shared library, rather than the static that find_package returns
  find_library(
      CUDART_LIB
      NAMES cudart
      PATHS
      ${CUDA_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES lib64 lib
      DOC "CUDA RT lib location"
      )

  set(CUDA_LIBRARIES "${CUDART_LIB};cuda")

  set(WITH_CUDA 1)
  set(OCCA_CUDA_ENABLED 1)
  include_directories( ${CUDA_INCLUDE_DIRS} )

  message(STATUS "CUDA version:       ${CUDA_VERSION_STRING}")
  message(STATUS "CUDA Include Paths: ${CUDA_INCLUDE_DIRS}")
  message(STATUS "CUDA Libraries:     ${CUDA_LIBRARIES}")
else (CUDA_FOUND)
  set(WITH_CUDA 0)
  set(OCCA_CUDA_ENABLED 0)
endif(CUDA_FOUND)
