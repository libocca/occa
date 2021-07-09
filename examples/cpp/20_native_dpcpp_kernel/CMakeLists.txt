if (OCCA_DPCPP_ENABLED)
  compile_cpp_example(native_dpcpp_kernel main.cpp)

  add_custom_target(cpp_example_native_dpcpp_kernel ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.cpp addVectors.cpp)
  add_dependencies(examples_cpp_native_dpcpp_kernel cpp_example_native_dpcpp_kernel)
endif()
