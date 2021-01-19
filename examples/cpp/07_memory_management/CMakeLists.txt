compile_cpp_example_with_modes(memory_management main.cpp)

add_custom_target(cpp_example_memory_management_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_memory_management cpp_example_memory_management_okl)
