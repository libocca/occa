if (WITH_CUDA)
  compile_cpp_example(cuda_interop main.cpp)

  add_custom_target(cpp_example_cuda_interop_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_cuda_interop cpp_example_cuda_interop_okl)
endif()
