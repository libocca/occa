###############################################################################
# FIND: HIP and associated helper binaries
###############################################################################
# HIP is supported on Linux only
if(UNIX AND NOT APPLE AND NOT CYGWIN)
  # Search for HIP installation
  if(NOT HIP_ROOT_DIR)
    # Search in user specified path first
    find_path(
      HIP_ROOT_DIR
      NAMES hipconfig
      PATHS
      ENV ROCM_PATH
      ENV HIP_PATH
      PATH_SUFFIXES bin
      DOC "HIP installed location"
      NO_DEFAULT_PATH
      )
    # Now search in default path
    find_path(
      HIP_ROOT_DIR
      NAMES hipconfig
      PATHS
      /opt/rocm
      /opt/rocm/hip
      PATH_SUFFIXES bin
      DOC "HIP installed location"
      )

    # Check if we found HIP installation
    if(HIP_ROOT_DIR)
      # If so, fix the path
      string(REGEX REPLACE "[/\\\\]?bin[64]*[/\\\\]?$" "" HIP_ROOT_DIR ${HIP_ROOT_DIR})
      # And push it back to the cache
      set(HIP_ROOT_DIR ${HIP_ROOT_DIR} CACHE PATH "HIP installed location" FORCE)
    endif()

    if(NOT EXISTS ${HIP_ROOT_DIR})
      if(HIP_FIND_REQUIRED)
        message(FATAL_ERROR "Specify HIP_ROOT_DIR")
      elseif(NOT HIP_FIND_QUIETLY)
        message("HIP_ROOT_DIR not found or specified")
      endif()
    endif()
  endif()

  # Find HIPCONFIG executable
  find_program(
    HIP_HIPCONFIG_EXECUTABLE
    NAMES hipconfig
    PATHS
    "${HIP_ROOT_DIR}"
    ENV ROCM_PATH
    ENV HIP_PATH
    /opt/rocm
    /opt/rocm/hip
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
    )
  if(NOT HIP_HIPCONFIG_EXECUTABLE)
    # Now search in default paths
    find_program(HIP_HIPCONFIG_EXECUTABLE hipconfig)
  endif()
  mark_as_advanced(HIP_HIPCONFIG_EXECUTABLE)

  if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_VERSION)
    # Compute the version
    execute_process(
      COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --version
      OUTPUT_VARIABLE _hip_version
      ERROR_VARIABLE _hip_error
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      )
    if(NOT _hip_error)
      set(HIP_VERSION ${_hip_version} CACHE STRING "Version of HIP as computed from hipcc")
    else()
      set(HIP_VERSION "0.0.0" CACHE STRING "Version of HIP as computed by FindHIP()")
    endif()
    mark_as_advanced(HIP_VERSION)
  endif()
  if(HIP_VERSION)
    string(REPLACE "." ";" _hip_version_list "${HIP_VERSION}")
    list(GET _hip_version_list 0 HIP_VERSION_MAJOR)
    list(GET _hip_version_list 1 HIP_VERSION_MINOR)
    list(GET _hip_version_list 2 HIP_VERSION_PATCH)
    set(HIP_VERSION_STRING "${HIP_VERSION}")
  endif()

  if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_PLATFORM)
    # Compute the platform
    execute_process(
      COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --platform
      OUTPUT_VARIABLE _hip_platform
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    set(HIP_PLATFORM ${_hip_platform} CACHE STRING "HIP platform as computed by hipconfig")
    mark_as_advanced(HIP_PLATFORM)
  endif()

  if(${HIP_PLATFORM} STREQUAL "hcc")
    set(HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include;${HIP_ROOT_DIR}/hcc/include")
    set(HIP_LIBRARIES "${HIP_ROOT_DIR}/lib/libhip_hcc.so")
    set(HIP_RUNTIME_DEFINE "__HIP_PLATFORM_HCC__")
  elseif(${HIP_PLATFORM} STREQUAL "nvcc")
    find_package(CUDA)

    #find the shared library, rather than the static that find_package returns
    find_library(
      CUDART_LIB
      NAMES cudart
      PATHS
      ${CUDA_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES lib64 lib
      DOC "CUDA RT lib location"
      )

    set(HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include;${CUDA_INCLUDE_DIRS}")
    set(HIP_LIBRARIES "${CUDART_LIB};cuda")
    set(HIP_RUNTIME_DEFINE "__HIP_PLATFORM_NVCC__")
  endif()
  mark_as_advanced(HIP_INCLUDE_DIRS)
  mark_as_advanced(HIP_LIBRARIES)
  mark_as_advanced(HIP_RUNTIME_DEFINE)

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    HIP
    REQUIRED_VARS
    HIP_ROOT_DIR
    HIP_INCLUDE_DIRS
    HIP_LIBRARIES
    HIP_RUNTIME_DEFINE
    HIP_HIPCONFIG_EXECUTABLE
    HIP_PLATFORM
    VERSION_VAR HIP_VERSION
    )