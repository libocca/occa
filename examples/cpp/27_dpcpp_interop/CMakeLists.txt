if (OCCA_DPCPP_ENABLED)
  compile_cpp_example(dpcpp_interop main.cpp)

  target_include_directories(examples_cpp_dpcpp_interop PRIVATE ${SYCL_INCLUDE_DIRS})
  target_link_libraries(examples_cpp_dpcpp_interop ${SYCL_LIBRARIES})
  target_compile_options(examples_cpp_dpcpp_interop BEFORE PRIVATE "${SYCL_CXX_FLAGS}")

  add_custom_target(cpp_example_dpcpp_interop ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_dpcpp_interop cpp_example_dpcpp_interop)
endif()
