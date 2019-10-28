find_package(CUDA)

IF (CUDA_FOUND)
    SET(OCCA_CUDA_ENABLED 1)
    SET(WITH_CUDA 1)
    include_directories( ${CUDA_INCLUDE_DIRS} )
ENDIF()
