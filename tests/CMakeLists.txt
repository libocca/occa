# Test occa binary (based on run_bin_tests)

add_test(NAME occa-autocomplete            COMMAND occa autocomplete)
add_test(NAME occa-clear                   COMMAND occa clear)
add_test(NAME occa-compile-help            COMMAND occa compile --help)
add_test(NAME occa-env                     COMMAND occa env)
add_test(NAME occa-info                    COMMAND occa info)
add_test(NAME occa-modes                   COMMAND occa modes)
add_test(NAME occa-translate-help          COMMAND occa translate --help)
add_test(NAME occa-translate-serial        COMMAND occa translate --mode Serial ${CMAKE_BINARY_DIR}/examples/cpp/01_add_vectors/addVectors.okl)
if (WITH_CUDA)
  add_test(NAME occa-translate-cuda          COMMAND occa translate --mode CUDA ${CMAKE_BINARY_DIR}/examples/cpp/01_add_vectors/addVectors.okl)
  add_test(NAME occa-translate-cuda-launcher COMMAND occa translate --mode CUDA --launcher ${CMAKE_BINARY_DIR}/examples/cpp/01_add_vectors/addVectors.okl)
endif()
add_test(NAME occa-version                 COMMAND occa version)

set_property(TEST occa-autocomplete occa-clear occa-compile-help occa-env occa-info occa-modes occa-translate-help occa-translate-serial occa-version APPEND PROPERTY ENVIRONMENT OCCA_CACHE_DIR=${CMAKE_BINARY_DIR}/cache)
if (WITH_CUDA)
  set_property(TEST occa-translate-cuda occa-translate-cuda-launcher APPEND PROPERTY ENVIRONMENT OCCA_CACHE_DIR=${CMAKE_BINARY_DIR}/cache)
endif()

add_custom_target(testinput ALL)
add_custom_command(TARGET testinput POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}/include")
add_custom_command(TARGET testinput POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/tests/files" "${CMAKE_BINARY_DIR}/tests/files")

add_subdirectory(src)
