if (OCCA_METAL_ENABLED)
  target_sources(libocca PRIVATE buffer.mm commandQueue.mm device.mm
    event.mm function.mm)
endif()
target_sources(libocca PRIVATE polyfill.cpp)
