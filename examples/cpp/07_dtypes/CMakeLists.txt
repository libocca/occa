COMPILE_CPP_EXAMPLE(07_dtypes main.cpp)

ADD_CUSTOM_TARGET(cpp_example_07_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
ADD_DEPENDENCIES(examples_cpp_07_dtypes cpp_example_07_addVectors_okl)
