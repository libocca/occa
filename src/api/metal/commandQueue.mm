#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/api/metal/commandQueue.hpp>

namespace occa {
  namespace api {
    namespace metal {
      commandQueue_t::commandQueue_t(void *obj_) :
      obj(obj_) {}

      commandQueue_t::commandQueue_t(const commandQueue_t &other) :
      obj(other.obj) {}

      void commandQueue_t::free() {
        if (obj) {
          // Remove reference count
          id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>) obj;
          commandQueue = nil;
          obj = NULL;
        }
      }
    }
  }
}

#endif
