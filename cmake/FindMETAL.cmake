###############################################################################
# FIND module for METAL components
# This Find module is also distributed alongside the occa package config file!
###############################################################################

find_library(METAL_LIBRARY Metal)
find_library(CORE_SERVICES CoreServices)
find_library(APP_KIT AppKit)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    METAL
    REQUIRED_VARS
    METAL_LIBRARY
    CORE_SERVICES
    APP_KIT
    )

if(METAL_FOUND AND NOT TARGET OCCA::depends::METAL)
  # Create our wrapper imported target
  # Put it in the OCCA namespace to make it clear that we created it.
  add_library(OCCA::depends::METAL INTERFACE IMPORTED)
  set_target_properties(OCCA::depends::METAL PROPERTIES
    INTERFACE_LINK_LIBRARIES " ${METAL_LIBRARY} ${CORE_SERVICES} ${APP_KIT}"
  )
endif()
