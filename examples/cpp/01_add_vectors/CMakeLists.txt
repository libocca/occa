compile_cpp_example_with_modes(01_add_vectors main.cpp)

add_custom_target(cpp_example_01_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_01_add_vectors cpp_example_01_addVectors_okl)
