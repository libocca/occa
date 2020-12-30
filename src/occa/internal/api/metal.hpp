#ifndef OCCA_INTERNAL_API_METAL_HEADER
#define OCCA_INTERNAL_API_METAL_HEADER

#include <occa/internal/api/metal/buffer.hpp>
#include <occa/internal/api/metal/commandQueue.hpp>
#include <occa/internal/api/metal/device.hpp>
#include <occa/internal/api/metal/event.hpp>
#include <occa/internal/api/metal/function.hpp>

namespace occa {
  namespace api {
    namespace metal {
      int getDeviceCount();

      device_t getDevice(const int deviceId);
    }
  }
}

#endif
