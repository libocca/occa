if (WITH_OPENMP)
  compile_cpp_example(15_openmp_interop main.cpp)

  add_custom_target(cpp_example_15_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_15_openmp_interop cpp_example_15_addVectors_okl)
endif (WITH_OPENMP)
