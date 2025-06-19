###############################################################################
# FIND module for METAL components
# This Find module is also distributed alongside the occa package config file!
###############################################################################

find_library(FOUNDATION_LIBRARY Foundation)
find_library(METAL_LIBRARY Metal)
find_library(METAL_KIT_LIBRARY MetalKit)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    METAL
    REQUIRED_VARS
    FOUNDATION_LIBRARY
    METAL_LIBRARY
    METAL_KIT_LIBRARY
)

if(METAL_FOUND AND NOT TARGET OCCA::depends::METAL)
  # Create our wrapper imported target
  # Put it in the OCCA namespace to make it clear that we created it.
  add_library(OCCA::depends::METAL INTERFACE IMPORTED)
  target_link_libraries(OCCA::depends::METAL INTERFACE
    ${FOUNDATION_LIBRARY} ${METAL_LIBRARY} ${METAL_KIT_LIBRARY}
  )
endif()
