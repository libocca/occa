compile_fortran_example_with_modes(01_add_vectors main.f90)

add_custom_target(fortran_example_01_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_fortran_01_add_vectors fortran_example_01_addVectors_okl)
