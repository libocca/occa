compile_c_example(reduction main.c)

add_custom_target(c_example_reduction_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/reduction.okl reduction.okl)
add_dependencies(examples_c_reduction c_example_reduction_okl)
