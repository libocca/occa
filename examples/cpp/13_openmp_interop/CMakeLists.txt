if (WITH_OPENMP)
  compile_cpp_example(openmp_interop main.cpp)

  add_custom_target(cpp_example_openmp_interop_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_openmp_interop cpp_example_openmp_interop_okl)
endif (WITH_OPENMP)
