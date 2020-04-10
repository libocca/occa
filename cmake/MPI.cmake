find_package(MPI)

if (MPI_FOUND)
    set(OCCA_MPI_ENABLED 1)
    set(WITH_MPI 1)
endif()
