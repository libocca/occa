#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Foundation/NSString.h>
#import <Metal/Metal.h>

#include <occa/internal/api/metal/device.hpp>
#include <occa/internal/io/lock.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace api {
    namespace metal {
      device_t::device_t(void *deviceObj_) :
        deviceObj(deviceObj_) {}

      device_t::device_t(const device_t &other) :
        deviceObj(other.deviceObj) {}

      device_t& device_t::operator = (const device_t &other) {
        deviceObj = other.deviceObj;
        return *this;
      }

      void device_t::free() {
        // Remove reference counts
        if (deviceObj) {
          id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;
          metalDevice = nil;
          deviceObj = NULL;
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

      function_t device_t::buildKernel(const std::string &metallibFilename,
                                       const std::string &kernelName,
                                       io::lock_t &lock) const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;

        NSString *metallibFilenameObj = [
          NSString stringWithCString:metallibFilename.c_str()
                   encoding:[NSString defaultCStringEncoding]
        ];
        NSString *kernelNameObj = [
          NSString stringWithCString:kernelName.c_str()
                   encoding:[NSString defaultCStringEncoding]
        ];

        NSError* error = nil;
        id<MTLLibrary> metalLibrary = [
          metalDevice newLibraryWithFile:metallibFilenameObj
                      error:&error
        ];

        if (!metalLibrary) {
          // An error occured building the library
          lock.release();
          if (error) {
            std::string errorStr = [error.localizedDescription UTF8String];
            OCCA_FORCE_ERROR("Device: Unable to create library from ["
                             << metallibFilename << "]."
                             << " Error: " << errorStr);
          } else {
            OCCA_FORCE_ERROR("Device: Unable to create library from ["
                             << metallibFilename << "].");
          }
          return function_t();
        }

        id<MTLFunction> metalFunction = [metalLibrary newFunctionWithName:kernelNameObj];

        if (!metalFunction) {
          // An error occured fetching the function from the library
          lock.release();
          OCCA_FORCE_ERROR("Device: Unable to get kernel ["
                           << kernelName << "] from library ["
                           << metallibFilename << "]");
          return function_t();
        }

        return function_t(const_cast<device_t*>(this),
                          (__bridge void*) metalLibrary,
                          (__bridge void*) metalFunction);
      }

      buffer_t device_t::malloc(const udim_t bytes,
                                const void *src) const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) deviceObj;

        id<MTLBuffer> buffer = (
          [metalDevice newBufferWithLength:bytes
                                   options:MTLResourceStorageModeManaged]
        );

        return buffer_t((__bridge void*) buffer);
      }
    }
  }
}

#endif
