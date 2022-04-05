compile_cpp_example_with_modes(async-build-kernels main.cpp)

add_custom_target(cpp_example_async-build-kernels_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_async-build-kernels cpp_example_async-build-kernels_okl)
