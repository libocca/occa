# Create
#   occaTargets.cmake, that defines IMPORTED targets for all targets associated with occaExport
#   occaConfig.cmake, used by find_package(occa), will use occaTargets.cmake to create the IMPORTED targets
#   occaConfigVersion.cmake, the version file associated with occaConfig.cmake

# Install in subdirectory lib/cmake/PACKAGENAME, which is where cmake expects package config files
set(PackageConfigInstallLocation lib/cmake/OCCA)
set(ExportNamespace "OCCA::")

# Set the exportPackageDependencies variable, for use in configuring occaConfig.cmake.in
# Do this for all our dependencies. In theory, could skip some if they are
# e.g. static or header libraries only AND we are only building shared libraries or executables AND
# we link PRIVATE, however the first condition is not easily checked.
set (exportPackageDependencies "")
string(APPEND exportPackageDependencies "find_dependency(Threads)\n")
if (OCCA_OPENMP_ENABLED)
  string(APPEND exportPackageDependencies "find_dependency(OpenMP)\n")
endif()
if (OCCA_CUDA_ENABLED)
  string(APPEND exportPackageDependencies "find_dependency(CUDAToolkit)\n")
endif()
if (OCCA_OPENCL_ENABLED)
  string(APPEND exportPackageDependencies "find_dependency(OpenCLWrapper)\n")
endif()
if(OCCA_DPCPP_ENABLED)
  string(APPEND exportPackageDependencies "find_dependency(DPCPP)\n")
endif()
if(OCCA_HIP_ENABLED)
  string(APPEND exportPackageDependencies "find_dependency(HIP)\n")
endif()
if(OCCA_METAL_ENABLED)
  string(APPEND exportPackageDependencies "find_dependency(METAL)\n")
endif()

# List of what targets are exported, for use in configuring occaConfig.cmake.in
# Explicit list because unfortunately no easy way to retrieve it through cmake, even though they all are part of the EXPORT occaExport
set(exportTargets "")
string(APPEND exportTargets "# ${ExportNamespace}libocca Target to link to for using occa\n")
string(APPEND exportTargets "# ${ExportNamespace}occa The occa executable, e.g. can be called to get information on supported backends\n")

include(CMakePackageConfigHelpers)
# Create the PackageConfig file, based on the template
configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/OCCAConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/OCCAConfig.cmake"
  INSTALL_DESTINATION ${PackageConfigInstallLocation} # Only used as relative reference during in this function, does not determine actual install location
  NO_CHECK_REQUIRED_COMPONENTS_MACRO # As long as components are not used, don't need it
  )

# Create the Version file
# Take the version string from occa.hpp
# Improvement possible: Put in occa.hpp, instead of read from. Requires that also non-cmake workflow supports that.
file(READ "${OCCA_SOURCE_DIR}/include/occa/defines/occa.hpp" occadefs)
string(REGEX MATCH "#define OCCA_VERSION_STR +\"([.0-9]*)\"" _ ${occadefs})
set(OCCA_VERSION_STR ${CMAKE_MATCH_1})
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/OCCAConfigVersion.cmake"
  VERSION "${OCCA_VERSION_STR}"
  COMPATIBILITY AnyNewerVersion
)

# Install the exported-targets file
# Will be used by the PackageConfig to generate the imported targets
install(
  EXPORT occaExport
  NAMESPACE ${ExportNamespace}
  FILE OCCATargets.cmake
  DESTINATION ${PackageConfigInstallLocation}
)

# Install the Config and Version files, and any files they need
# Note that find_package is case-sensitive w.r.t. the name of the package as part of the name of these files
#   find_package(occa) will only match occaConfig.cmake, NOT OccaConfig.cmake
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OCCAConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/OCCAConfigVersion.cmake"
    "${CMAKE_CURRENT_LIST_DIR}/FindDPCPP.cmake"
    "${CMAKE_CURRENT_LIST_DIR}/FindHIP.cmake"
    "${CMAKE_CURRENT_LIST_DIR}/FindMETAL.cmake"
    "${CMAKE_CURRENT_LIST_DIR}/FindOpenCLWrapper.cmake"
  DESTINATION
    ${PackageConfigInstallLocation}
)