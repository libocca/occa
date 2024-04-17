compile_cpp_example_with_modes(stream_tags main.cpp)

add_custom_target(cpp_example_stream_tags_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/powerOfPi2.okl powerOfPi2.okl)
add_dependencies(examples_cpp_stream_tags cpp_example_stream_tags_okl)
