compile_fortran_example_with_modes(04_reduction main.f90)

set_source_files_properties(main.f90 PROPERTIES COMPILE_FLAGS "${SUPPORTED_WNO_INTEGER_DIVISION_Fortran_FLAGS}")

add_custom_target(fortran_example_04_reduction_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/reduction.okl reduction.okl)
add_dependencies(examples_fortran_04_reduction fortran_example_04_reduction_okl)
