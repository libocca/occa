set(EXAMPLE_NAME "device_function")
compile_cpp_example_with_modes(${EXAMPLE_NAME} main.cpp)

add_custom_target(cpp_example_${EXAMPLE_NAME}_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_cpp_${EXAMPLE_NAME} cpp_example_${EXAMPLE_NAME}_okl)
