#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/internal/api/metal/buffer.hpp>

namespace occa {
  namespace api {
    namespace metal {
      buffer_t::buffer_t(void *bufferObj_) :
        bufferObj(bufferObj_),
        ptr(NULL) {}

      buffer_t::buffer_t(const buffer_t &other) :
        bufferObj(other.bufferObj),
        ptr(other.ptr) {}

      buffer_t& buffer_t::operator = (const buffer_t &other) {
        bufferObj = other.bufferObj;
        ptr = other.ptr;
        return *this;
      }

      void buffer_t::free() {
        // Remove reference count
        if (bufferObj) {
          id<MTLBuffer> metalBuffer = (__bridge id<MTLBuffer>) bufferObj;
          metalBuffer = nil;
          bufferObj = NULL;
          ptr = NULL;
        }
      }

      void* buffer_t::getPtr() const {
        if (!ptr) {
          id<MTLBuffer> metalBuffer = (__bridge id<MTLBuffer>) bufferObj;
          ptr = (__bridge void*) metalBuffer.contents;
        }
        return ptr;
      }
    }
  }
}

#endif
