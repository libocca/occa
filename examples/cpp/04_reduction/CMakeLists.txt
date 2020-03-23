compile_cpp_example_with_modes(04_reduction main.cpp)

add_custom_target(cpp_example_04_reduction_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/reduction.okl reduction.okl)
add_dependencies(examples_cpp_04_reduction cpp_example_04_reduction_okl)
