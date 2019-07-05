#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/api/metal/buffer.hpp>

namespace occa {
  namespace api {
    namespace metal {
      buffer_t::buffer_t(void *obj_) :
        obj(obj_),
        ptr(NULL) {}

      buffer_t::buffer_t(const buffer_t &other) :
        obj(other.obj),
        ptr(other.ptr) {}

      void buffer_t::free() {
        if (obj) {
          // Remove reference count
          id<MTLBuffer> buffer = (__bridge id<MTLBuffer>) obj;
          buffer = nil;
          obj = NULL;
          ptr = NULL;
        }
      }

      void* buffer_t::getPtr() const {
        if (!ptr) {
          id<MTLBuffer> buffer = (__bridge id<MTLBuffer>) obj;
          ptr = (__bridge void*) buffer.contents;
        }
        return ptr;
      }
    }
  }
}

#endif
