compile_cpp_example_with_modes(nonblocking_streams main.cpp)

add_custom_target(cpp_example_nonblocking_streams_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/powerOfPi.okl powerOfPi.okl)
add_dependencies(examples_cpp_nonblocking_streams cpp_example_nonblocking_streams_okl)
