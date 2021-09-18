compile_cpp_example(native_c_kernels main.cpp)

add_custom_target(cpp_example_native_c_kernels_c ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.c addVectors.c)
add_dependencies(examples_cpp_native_c_kernels cpp_example_native_c_kernels_c)
