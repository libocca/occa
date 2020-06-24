if (MPI_FOUND)
  compile_fortran_mpi_example_with_modes(10_mpi main.f90)

  add_custom_target(fortran_example_10_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_fortran_10_mpi fortran_example_10_addVectors_okl)
endif()
