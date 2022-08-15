function(add_occa_test test_source)
  # Metadata
  get_filename_component(source_directory ${test_source} DIRECTORY)
  get_filename_component(test_name ${test_source} NAME_WLE)

  string(REGEX REPLACE "src/" "bin/" test_directory "${source_directory}")

  set(test_binary "${test_directory}/${test_name}")

  string(REGEX REPLACE "/" "-" cmake_test_target "${test_binary}")

  # Setup executable target
  add_executable(${cmake_test_target} ${test_source})

  set_target_properties(${cmake_test_target} PROPERTIES
    OUTPUT_NAME ${test_name}
    RUNTIME_OUTPUT_DIRECTORY ${test_directory})

  # Build config
  target_link_libraries(${cmake_test_target} libocca ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
  target_include_directories(${cmake_test_target} PRIVATE
    $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)

  if (${test_source} MATCHES "typedefs.f90")
    target_link_libraries(${cmake_test_target} libtypedefs_helper)
  endif()

  add_test(
    NAME ${cmake_test_target}
    COMMAND ${test_binary})

  # Expected defines
  set_property(TEST ${cmake_test_target} APPEND PROPERTY
    ENVIRONMENT OCCA_CACHE_DIR=${OCCA_BUILD_DIR}/occa)
endfunction()

#---[ Setup Tests ]---------------------
file(
  GLOB_RECURSE occa_tests
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/*.cpp")

if (ENABLE_FORTRAN)
  file(
    GLOB_RECURSE occa_fortran_tests
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/*.f90")

  list(APPEND occa_tests ${occa_fortran_tests})

  list(FILTER occa_tests
    EXCLUDE REGEX "src/fortran/typedefs_helper.cpp")

  add_library(libtypedefs_helper SHARED "src/fortran/typedefs_helper.cpp")
  target_link_libraries(libtypedefs_helper libocca)
else()
  list(FILTER occa_tests
    EXCLUDE REGEX "src/fortran")
endif()

foreach(occa_test ${occa_tests})
  add_occa_test("${occa_test}")
endforeach()
#=======================================
