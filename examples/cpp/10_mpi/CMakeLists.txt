if (MPI_FOUND)
  add_executable(examples_cpp_10_mpi main.cpp)
  target_link_libraries(examples_cpp_10_mpi libocca)

  add_custom_target(cpp_example_10_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_10_mpi cpp_example_10_addVectors_okl)

  add_test(NAME examples_cpp_10_mpi COMMAND ${MPIEXEC_EXECUTABLE} -np 2 examples_cpp_10_mpi --verbose)
  set_property(TEST examples_cpp_10_mpi APPEND PROPERTY ENVIRONMENT OCCA_CACHE_DIR=${CMAKE_BINARY_DIR}/occa)
endif()
