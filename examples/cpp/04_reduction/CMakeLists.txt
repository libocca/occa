COMPILE_CPP_EXAMPLE(04_reduction main.cpp)

ADD_CUSTOM_TARGET(cpp_example_04_reduction_okl ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/reduction.okl reduction.okl)
ADD_DEPENDENCIES(examples_cpp_04_reduction cpp_example_04_reduction_okl)
