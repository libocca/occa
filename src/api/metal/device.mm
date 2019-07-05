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
        if (deviceObj) {
          // Remove reference count
          id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
          dev = nil;
          deviceObj = NULL;
        }

        if (libraryObj) {
          // Remove reference count
          id<MTLLibrary> dev = (__bridge id<MTLLibrary>) libraryObj;
          dev = nil;
          libraryObj = NULL;
        }
      }

      std::string device_t::getName() const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
        return [dev.name UTF8String];
      }

      udim_t device_t::getMemorySize() const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
        return [dev recommendedMaxWorkingSetSize];
      }

      dim device_t::getMaxOuterDims() const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
        return dim(-1, -1, -1);
      }

      dim device_t::getMaxInnerDims() const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
        MTLSize size = dev.maxThreadsPerThreadgroup;
        return dim(size.width, size.height, size.depth);
      }

      commandQueue_t device_t::createCommandQueue() const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;

        id<MTLCommandQueue> commandQueue = [dev newCommandQueue];
        void *commandQueueObj = (__bridge void*) commandQueue;

        return commandQueue_t(commandQueueObj);
      }

      event_t device_t::createEvent() const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;

        id<MTLEvent> event = [dev newEvent];
        void *eventObj = (__bridge void*) event;

        return event_t(eventObj);
      }

      kernel_t device_t::buildKernel(const std::string &source,
                                     const std::string &kernelName,
                                     io::lock_t &lock) const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
        return kernel_t();
      }

      buffer_t device_t::malloc(const udim_t bytes,
                                const void *src) const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;

        id<MTLBuffer> buffer = (
          [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]
        );

        return buffer_t();
      }

      void device_t::memcpy(buffer_t &dest,
                            const udim_t destOffset,
                            const buffer_t &src,
                            const udim_t srcOffset,
                            const udim_t bytes,
                            const bool async) const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
      }

      void device_t::memcpy(void *dest,
                            const buffer_t &src,
                            const udim_t srcOffset,
                            const udim_t bytes,
                            const bool async) const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
      }

      void device_t::memcpy(buffer_t &dest,
                            const udim_t destOffset,
                            const void *src,
                            const udim_t bytes,
                            const bool async) const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
      }

      void device_t::waitFor(event_t &event) const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
      }

      void device_t::device_t::finish() const {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) deviceObj;
      }
    }
  }
}

#endif
