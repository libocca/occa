#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Foundation/NSArray.h>
#import <Metal/Metal.h>

#include <occa/internal/api/metal.hpp>

namespace occa {
  namespace api {
    namespace metal {
      int getDeviceCount() {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        return [devices count];
      }

      device_t getDevice(const int deviceId) {
        NSArray <id<MTLDevice>> *devices = MTLCopyAllDevices();

        id<MTLDevice> device = devices[deviceId];
        void *deviceObj = (__bridge void*) device;

        return device_t(deviceObj);
      }
    }
  }
}

#endif
