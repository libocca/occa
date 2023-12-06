## Generated codes for handling kernel arguments
#    Configure default/pre-generated files
#    Check if those files need to be re-generated
option(KEEP_DEFAULT_MAX_NUM_INLINE_KERNEL_ARGS "Skip code generations for the given maximum number of arguments on inline kernels, use default instead" OFF) 
set(MAX_NUM_KERNEL_ARGS_DEFAULT "128")
set(MAX_NUM_KERNEL_ARGS ${MAX_NUM_KERNEL_ARGS_DEFAULT} CACHE STRING "The maximum number of allowed kernel arguments")
set(OCCA_MAX_ARGS ${MAX_NUM_KERNEL_ARGS_DEFAULT} CACHE STRING "The maximum number of allowed kernel arguments stored to use, do not change this value directly, update MAX_NUM_KERNEL_ARGS instead")
if (NOT ${MAX_NUM_KERNEL_ARGS} EQUAL ${OCCA_MAX_ARGS})
  execute_process(COMMAND python --version OUTPUT_VARIABLE python_version)
  string(REGEX MATCH "[0-9.]\+" python_version ${python_version})
  if ("${python_version}" VERSION_LESS "3.7.2")
    message(WARNING "-- Failed to set the maximum number of kernel arguments to ${MAX_NUM_KERNEL_ARGS}, required minimum python version 3.7.2. The default value ${MAX_NUM_KERNEL_ARGS_DEFAULT} will be used.")
  else()
    message("-- Codegen for the maximum number of kernel arguments : ${MAX_NUM_KERNEL_ARGS}")
    if (KEEP_DEFAULT_MAX_NUM_INLINE_KERNEL_ARGS AND ${MAX_NUM_KERNEL_ARGS} GREATER ${MAX_NUM_KERNEL_ARGS_DEFAULT})
      execute_process(COMMAND ${CMAKE_COMMAND} -E env OCCA_DIR=${OCCA_BUILD_DIR} python ${OCCA_SOURCE_DIR}/scripts/codegen/setup_kernel_operators.py -N ${MAX_NUM_KERNEL_ARGS} --skipInline)
    else()
      execute_process(COMMAND ${CMAKE_COMMAND} -E env OCCA_DIR=${OCCA_BUILD_DIR} python ${OCCA_SOURCE_DIR}/scripts/codegen/setup_kernel_operators.py -N ${MAX_NUM_KERNEL_ARGS})
    endif()
    set(OCCA_MAX_ARGS ${MAX_NUM_KERNEL_ARGS} CACHE STRING "The maximum number of allowed kernel arguments stored to use, do not change this value directly, update MAX_NUM_KERNEL_ARGS instead" FORCE)
  endif()
else()
  if (${OCCA_MAX_ARGS} EQUAL ${MAX_NUM_KERNEL_ARGS_DEFAULT})
    configure_file(${OCCA_SOURCE_DIR}/scripts/codegen/kernelOperators.hpp_codegen.in ${OCCA_BUILD_DIR}/include/codegen/kernelOperators.hpp_codegen COPYONLY)
    configure_file(${OCCA_SOURCE_DIR}/scripts/codegen/kernelOperators.cpp_codegen.in ${OCCA_BUILD_DIR}/include/codegen/kernelOperators.cpp_codegen COPYONLY)
    configure_file(${OCCA_SOURCE_DIR}/scripts/codegen/runFunction.cpp_codegen.in ${OCCA_BUILD_DIR}/include/codegen/runFunction.cpp_codegen COPYONLY)
    configure_file(${OCCA_SOURCE_DIR}/scripts/codegen/macros.hpp_codegen.in ${OCCA_BUILD_DIR}/include/codegen/macros.hpp_codegen COPYONLY)
  endif()
endif()

#    Set installtion of files required in header
install(
  FILES ${OCCA_BUILD_DIR}/include/codegen/kernelOperators.hpp_codegen
  DESTINATION include/occa/core/codegen)
install(
  FILES ${OCCA_BUILD_DIR}/include/codegen/macros.hpp_codegen
  DESTINATION include/occa/defines/codegen)
