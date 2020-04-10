if (WITH_CUDA)
  compile_cpp_example(16_cuda_interop main.cpp)

  add_custom_target(cpp_example_16_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_16_cuda_interop cpp_example_16_addVectors_okl)
endif()
