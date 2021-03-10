compile_cpp_example_with_modes(custom_types main.cpp)

add_custom_target(cpp_example_custom_types_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_custom_types cpp_example_custom_types_okl)
