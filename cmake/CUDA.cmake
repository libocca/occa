find_package(CUDA)

if (CUDA_FOUND)
    set(OCCA_CUDA_ENABLED 1)
    set(WITH_CUDA 1)
    include_directories( ${CUDA_INCLUDE_DIRS} )
endif()
