COMPILE_CPP_EXAMPLE(08_arrays main.cpp)

ADD_CUSTOM_TARGET(cpp_example_08_smallTranspose_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/smallTranspose.okl smallTranspose.okl)
ADD_DEPENDENCIES(examples_cpp_08_arrays cpp_example_08_smallTranspose_okl)
