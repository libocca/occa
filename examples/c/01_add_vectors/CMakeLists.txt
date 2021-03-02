compile_c_example(add_vector main.c)

add_custom_target(c_example_add_vectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_c_add_vector c_example_add_vectors_okl)
