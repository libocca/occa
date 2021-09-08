compile_cpp_example(native_cpp_kernels main.cpp)

add_custom_target(cpp_example_native_cpp_kernels_cpp ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.cpp addVectors.cpp)
add_dependencies(examples_cpp_native_cpp_kernels cpp_example_native_cpp_kernels_cpp)
