compile_cpp_example_with_modes(06_unified_memory main.cpp)

add_custom_target(cpp_example_06_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_06_unified_memory cpp_example_06_addVectors_okl)
