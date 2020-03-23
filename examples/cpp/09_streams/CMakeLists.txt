compile_cpp_example_with_modes(09_streams main.cpp)

add_custom_target(cpp_example_09_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_09_streams cpp_example_09_addVectors_okl)
