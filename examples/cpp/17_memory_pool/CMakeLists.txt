compile_cpp_example(memory_pool main.cpp)

add_custom_target(cpp_example_memory_pool_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_memory_pool cpp_example_memory_pool_okl)
