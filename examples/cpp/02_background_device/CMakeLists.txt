COMPILE_CPP_EXAMPLE(02_background_device main.cpp)

ADD_CUSTOM_TARGET(cpp_example_02_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
ADD_DEPENDENCIES(examples_cpp_02_background_device cpp_example_02_addVectors_okl)
