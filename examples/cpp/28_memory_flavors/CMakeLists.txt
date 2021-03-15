compile_cpp_example_with_modes(memory_flavors main.cpp)

add_custom_target(cpp_example_memory_flavors ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_memory_flavors cpp_example_memory_flavors)
