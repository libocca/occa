#ifndef OCCA_API_METAL_HEADER
#define OCCA_API_METAL_HEADER

#include <occa/api/metal/buffer.hpp>
#include <occa/api/metal/commandQueue.hpp>
#include <occa/api/metal/device.hpp>
#include <occa/api/metal/event.hpp>
#include <occa/api/metal/function.hpp>

namespace occa {
  namespace api {
    namespace metal {
      int getDeviceCount();

      device_t getDevice(const int deviceId);
    }
  }
}

#endif
