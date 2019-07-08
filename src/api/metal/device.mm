#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Foundation/NSString.h>
#import <Metal/Metal.h>

#include <occa/api/metal/device.hpp>

namespace occa {
  namespace api {
    namespace metal {
      device_t::device_t(void *deviceObj_) :
        deviceObj(deviceObj_) {}

      device_t::device_t(const device_t &other) :
        deviceObj(other.deviceObj),
        libraryObj(other.libraryObj) {}

      void device_t::free() {
        // Remove reference counts
        if (deviceObj) {
          id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;
          metalDevice = nil;
          deviceObj = NULL;
        }
        if (libraryObj) {
          id<MTLLibrary> metalLibrary = (__bridge id<MTLLibrary>) libraryObj;
          metalLibrary = nil;
          libraryObj = NULL;
        }
      }

      std::string device_t::getName() const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;
        return [metalDevice.name UTF8String];
      }

      udim_t device_t::getMemorySize() const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;
        return [metalDevice recommendedMaxWorkingSetSize];
      }

      dim device_t::getMaxOuterDims() const {
        return dim(-1, -1, -1);
      }

      dim device_t::getMaxInnerDims() const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;
        MTLSize size = metalDevice.maxThreadsPerThreadgroup;
        return dim(size.width, size.height, size.depth);
      }

      commandQueue_t device_t::createCommandQueue() const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;

        id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
        void *commandQueueObj = (__bridge void*) commandQueue;

        return commandQueue_t(const_cast<device_t*>(this),
                              commandQueueObj);
      }

      function_t device_t::buildKernel(const std::string &source,
                                       const std::string &kernelName,
                                       io::lock_t &lock) const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;

        // TODO

        return function_t();
      }

      buffer_t device_t::malloc(const udim_t bytes,
                                const void *src) const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;

        id<MTLBuffer> buffer = (
          [metalDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared]
        );

        return buffer_t((__bridge void*) buffer);
      }

      void device_t::memcpy(buffer_t &dest,
                            const udim_t destOffset,
                            const buffer_t &src,
                            const udim_t srcOffset,
                            const udim_t bytes,
                            const bool async) const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;
        // TODO
      }

      void device_t::memcpy(void *dest,
                            const buffer_t &src,
                            const udim_t srcOffset,
                            const udim_t bytes,
                            const bool async) const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;
        // TODO
      }

      void device_t::memcpy(buffer_t &dest,
                            const udim_t destOffset,
                            const void *src,
                            const udim_t bytes,
                            const bool async) const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;
        // TODO
      }
    }
  }
}

#endif
