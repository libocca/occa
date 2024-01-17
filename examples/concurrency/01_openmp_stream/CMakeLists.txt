compile_concurrency_example(openmp_stream main.cpp)

add_custom_target(concurrency_example_openmp_stream_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/powerOfPi.okl powerOfPi.okl)
add_dependencies(examples_concurrency_openmp_stream concurrency_example_openmp_stream_okl)
