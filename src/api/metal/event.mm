#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/api/metal/event.hpp>

namespace occa {
  namespace api {
    namespace metal {
      event_t::event_t() :
        eventObj(NULL),
        eventId(-1),
        commandBufferObj(NULL),
        eventTime(0) {}

        event_t::event_t(void *eventObj_,
                         const int eventId_,
                         void *commandBufferObj_) :
        eventObj(eventObj_),
        eventId(eventId_),
        commandBufferObj(commandBufferObj_),
        eventTime(0) {}

      event_t::event_t(const event_t &other) :
        eventId(other.eventId),
        eventObj(other.eventObj),
        commandBufferObj(other.commandBufferObj),
        eventTime(other.eventTime) {}

      void event_t::free() {
        // Remove reference count
        eventTime = 0;
        eventId = -1;
        if (eventObj) {
          id<MTLEvent> metalEvent = (__bridge id<MTLEvent>) eventObj;
          metalEvent = nil;
          eventObj = NULL;
        }
        freeCommandBuffer();
      }

      void event_t::freeCommandBuffer() {
        if (commandBufferObj) {
          id<MTLCommandBuffer> metalCommandBuffer = (
            (__bridge id<MTLCommandBuffer>) commandBufferObj
          );
          metalCommandBuffer = nil;
          commandBufferObj = NULL;
        }
      }

      void event_t::waitUntilCompleted() {
        if (commandBufferObj) {
          id<MTLCommandBuffer> metalCommandBuffer = (
            (__bridge id<MTLCommandBuffer>) commandBufferObj
          );
          [metalCommandBuffer waitUntilCompleted];
          freeCommandBuffer();
        }
      }

      void event_t::setTime(const double eventTime_) {
        freeCommandBuffer();
        eventTime = eventTime_;
      }

      double event_t::getTime() const {
        return eventTime;
      }
    }
  }
}

#endif
