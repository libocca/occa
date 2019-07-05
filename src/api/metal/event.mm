#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/api/metal/event.hpp>

namespace occa {
  namespace api {
    namespace metal {
      event_t::event_t(void *obj_) :
        obj(obj_) {}

      event_t::event_t(const event_t &other) :
        obj(other.obj) {}

      void event_t::free() {
        if (obj) {
          // Remove reference count
          id<MTLEvent> event = (__bridge id<MTLEvent>) obj;
          event = nil;
          obj = NULL;
        }
      }

      double event_t::getTime() const {
        return 0;
      }
    }
  }
}

#endif
