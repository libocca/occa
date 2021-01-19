compile_cpp_example_with_modes(add_vectors main.cpp)

add_custom_target(cpp_example_add_vectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_add_vectors cpp_example_add_vectors_okl)
