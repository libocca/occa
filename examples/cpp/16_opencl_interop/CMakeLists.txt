if (WITH_OPENCL)
  compile_cpp_example(opencl_interop main.cpp)

  add_custom_target(cpp_example_opencl_interop_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_opencl_interop cpp_example_opencl_interop_okl)
endif()
