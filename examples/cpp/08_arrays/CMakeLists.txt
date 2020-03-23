compile_cpp_example(08_arrays main.cpp)

add_custom_target(cpp_example_08_smallTranspose_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/smallTranspose.okl smallTranspose.okl)
add_dependencies(examples_cpp_08_arrays cpp_example_08_smallTranspose_okl)
