if (OCCA_METAL_ENABLED)
  target_sources(libocca PRIVATE metal.mm)
endif()
add_subdirectory(metal)
