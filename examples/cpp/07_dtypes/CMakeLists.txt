compile_cpp_example(07_dtypes main.cpp)

add_custom_target(cpp_example_07_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_07_dtypes cpp_example_07_addVectors_okl)
