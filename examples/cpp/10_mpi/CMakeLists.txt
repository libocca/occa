IF (MPI_FOUND)
    INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
    COMPILE_CPP_EXAMPLE(10_mpi main.cpp)

    ADD_CUSTOM_TARGET(cpp_example_10_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
    ADD_DEPENDENCIES(examples_cpp_10_mpi cpp_example_10_addVectors_okl)
ENDIF()
