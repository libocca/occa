COMPILE_CPP_EXAMPLE(06_unified_memory main.cpp)

ADD_CUSTOM_TARGET(cpp_example_06_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
ADD_DEPENDENCIES(examples_cpp_06_unified_memory cpp_example_06_addVectors_okl)
