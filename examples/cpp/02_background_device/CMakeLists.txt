compile_cpp_example(02_background_device main.cpp)

add_custom_target(cpp_example_02_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_02_background_device cpp_example_02_addVectors_okl)
