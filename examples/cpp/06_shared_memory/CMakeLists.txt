compile_cpp_example_with_modes(shared_memory main.cpp)

add_custom_target(cpp_example_shared_memory_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/reduction.okl reduction.okl)
add_dependencies(examples_cpp_shared_memory cpp_example_shared_memory_okl)
