#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/api/metal/commandQueue.hpp>

namespace occa {
  namespace api {
    namespace metal {
      commandQueue_t::commandQueue_t(void *commandQueueObj_) :
      commandQueueObj(commandQueueObj_) {}

      commandQueue_t::commandQueue_t(const commandQueue_t &other) :
      commandQueueObj(other.commandQueueObj) {}

      void commandQueue_t::free() {
          // Remove reference count
        if (commandQueueObj) {
          id<MTLCommandQueue> metalCommandQueue = (__bridge id<MTLCommandQueue>) commandQueueObj;
          metalCommandQueue = nil;
          commandQueueObj = NULL;
        }
      }
    }
  }
}

#endif
