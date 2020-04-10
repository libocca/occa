compile_c_example(01_add_vector main.c)

add_custom_target(c_example_01_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_c_01_add_vector c_example_01_addVectors_okl)
