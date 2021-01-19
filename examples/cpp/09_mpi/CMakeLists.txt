if (MPI_FOUND)
  compile_cpp_mpi_example_with_modes(mpi main.cpp)

  add_custom_target(cpp_example_mpi_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_mpi cpp_example_mpi_okl)
endif()
