compile_fortran_example_with_modes(09_streams main.f90)

add_custom_target(fortran_example_09_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_fortran_09_streams fortran_example_09_addVectors_okl)
