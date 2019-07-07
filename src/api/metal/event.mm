#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/api/metal/event.hpp>

namespace occa {
  namespace api {
    namespace metal {
      event_t::event_t(void *eventObj_) :
        eventObj(eventObj_) {}

      event_t::event_t(const event_t &other) :
        eventObj(other.eventObj) {}

      void event_t::free() {
        if (eventObj) {
          // Remove reference count
          id<MTLEvent> metalEvent = (__bridge id<MTLEvent>) eventObj;
          metalEvent = nil;
          eventObj = NULL;
        }
      }

      double event_t::getTime() const {
        // TODO
        return 0;
      }
    }
  }
}

#endif
