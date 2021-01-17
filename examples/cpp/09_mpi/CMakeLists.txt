if (MPI_FOUND)
  compile_cpp_mpi_example_with_modes(10_mpi main.cpp)

  add_custom_target(cpp_example_10_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_10_mpi cpp_example_10_addVectors_okl)
endif()
