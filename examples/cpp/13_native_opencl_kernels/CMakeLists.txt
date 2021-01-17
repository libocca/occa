if (WITH_OPENCL)
  compile_cpp_example(14_native_opencl_kernels main.cpp)

  add_custom_target(cpp_example_14_addVectors_cl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.cl addVectors.cl)
  add_dependencies(examples_cpp_14_native_opencl_kernels cpp_example_14_addVectors_cl)
endif()
