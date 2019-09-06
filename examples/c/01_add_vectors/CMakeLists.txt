COMPILE_C_EXAMPLE(01_add_vector main.c)

ADD_CUSTOM_TARGET(c_example_01_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
ADD_DEPENDENCIES(examples_c_01_add_vector c_example_01_addVectors_okl)
