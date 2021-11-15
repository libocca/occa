###############################################################################
# FIND: HIP and associated helper binaries
# This Find module is also distributed alongside the occa package config file!
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

  if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_COMPILER)
    # Find the compiler
    execute_process(
      COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --compiler
      OUTPUT_VARIABLE _hip_compiler
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    set(HIP_COMPILER ${_hip_compiler} CACHE STRING "HIP compiler as computed by hipconfig")
    mark_as_advanced(HIP_COMPILER)
  endif()

  if(HIP_PLATFORM)
    if(${HIP_PLATFORM} STREQUAL "hcc" OR ${HIP_PLATFORM} STREQUAL "amd")
      if(${HIP_COMPILER} STREQUAL "hcc")
        set(HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include;${HIP_ROOT_DIR}/hcc/include")
        set(HIP_LIBRARIES "${HIP_ROOT_DIR}/lib/libhip_hcc.so")
        set(HIP_RUNTIME_DEFINE "__HIP_PLATFORM_HCC__")
      elseif(${HIP_COMPILER} STREQUAL "clang")
        set(HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")
        set(HIP_LIBRARIES "${HIP_ROOT_DIR}/lib/libamdhip64.so")
        set(HIP_RUNTIME_DEFINE "__HIP_PLATFORM_HCC__")
        set(HIP_PLATFORM "hip-clang")
      endif()
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
    HIP_COMPILER
    VERSION_VAR HIP_VERSION
    )

if(HIP_FOUND AND NOT TARGET OCCA::depends::HIP)
  # Create our wrapper imported target
  # Put it in the OCCA namespace to make it clear that we created it.
  add_library(OCCA::depends::HIP INTERFACE IMPORTED)
  set_target_properties(OCCA::depends::HIP PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${HIP_RUNTIME_DEFINE}"
    INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${HIP_LIBRARIES}"
  )
endif()
