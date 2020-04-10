if (WITH_CUDA)
  compile_cpp_example(13_native_cuda_kernels main.cpp)

  add_custom_target(cpp_example_13_addVectors_cu ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.cu addVectors.cu)
  add_dependencies(examples_cpp_13_native_cuda_kernels cpp_example_13_addVectors_cu)
endif()
