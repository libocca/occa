COMPILE_C_EXAMPLE(04_reduction main.c)

ADD_CUSTOM_TARGET(c_example_04_reduction_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/reduction.okl reduction.okl)
ADD_DEPENDENCIES(examples_c_04_reduction c_example_04_reduction_okl)
