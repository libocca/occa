COMPILE_CPP_EXAMPLE(09_streams main.cpp)

ADD_CUSTOM_TARGET(cpp_example_09_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
ADD_DEPENDENCIES(examples_cpp_09_streams cpp_example_09_addVectors_okl)
