#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#include <occa/api/metal.hpp>

namespace occa {
  namespace api {
    namespace metal {
      int getDeviceCount() {
        return 0;
      }

      device_t getDevice(const int id) {
        return device_t();
      }
    }
  }
}

#endif
