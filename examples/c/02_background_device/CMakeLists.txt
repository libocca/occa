COMPILE_C_EXAMPLE(02_background_device main.c)

ADD_CUSTOM_TARGET(c_example_02_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
ADD_DEPENDENCIES(examples_c_02_background_device c_example_02_addVectors_okl)
