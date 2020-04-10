compile_cpp_example_with_modes(02_background_device main.cpp)

add_custom_target(cpp_example_02_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_02_background_device cpp_example_02_addVectors_okl)
