if (WITH_CUDA)
  compile_cpp_example(native_cuda_kernels main.cpp)

  add_custom_target(cpp_example_native_cuda_kernels_cu ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.cu addVectors.cu)
  add_dependencies(examples_cpp_native_cuda_kernels cpp_example_native_cuda_kernels_cu)
endif()
