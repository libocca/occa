#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/api/metal/commandQueue.hpp>
#include <occa/api/metal/device.hpp>
#include <occa/api/metal/event.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  namespace api {
    namespace metal {
      commandQueue_t::commandQueue_t() :
        device(NULL),
        commandQueueObj(NULL),
        lastCommandBufferObj(NULL),
        lastCommandId(0) {}

      commandQueue_t::commandQueue_t(device_t *device,
                                     void *commandQueueObj_) :
        device(device),
        commandQueueObj(commandQueueObj_),
        lastCommandBufferObj(NULL),
        lastCommandId(0) {}

      commandQueue_t::commandQueue_t(const commandQueue_t &other) :
        device(other.device),
        commandQueueObj(other.commandQueueObj),
        lastCommandBufferObj(other.lastCommandBufferObj),
        events(other.events),
        lastCommandId(other.lastCommandId) {}

      void commandQueue_t::free() {
        device = NULL;
        // Remove reference count
        if (commandQueueObj) {
          id<MTLCommandQueue> metalCommandQueue = (__bridge id<MTLCommandQueue>) commandQueueObj;
          metalCommandQueue = nil;
          commandQueueObj = NULL;
        }
        freeLastCommandBuffer();
        lastCommandId = 0;
        events.clear();
      }

      void commandQueue_t::freeLastCommandBuffer() {
        if (lastCommandBufferObj) {
          id<MTLCommandBuffer> metalCommandBuffer = (
            (__bridge id<MTLCommandBuffer>) lastCommandBufferObj
          );
          metalCommandBuffer = nil;
          lastCommandBufferObj = NULL;
        }
      }

      event_t commandQueue_t::createEvent() const {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) device->deviceObj;

        id<MTLEvent> event = [metalDevice newEvent];
        void *eventObj = (__bridge void*) event;

        return event_t(eventObj, lastCommandId, lastCommandBufferObj);
      }

      void commandQueue_t::setLastCommandBuffer(void *commandBufferObj) {
        if (!commandBufferObj) {
          return;
        }
        freeLastCommandBuffer();
        lastCommandBufferObj = commandBufferObj;

        const int eventId = ++lastCommandId;
        id<MTLCommandBuffer> metalCommandBuffer = (
          (__bridge id<MTLCommandBuffer>) lastCommandBufferObj
        );
        [metalCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            processEvents(eventId);
        }];
      }

      void commandQueue_t::processEvents(const int eventId) {
        /*
          KC: Kernel command
          BC: Buffer command
          E#: Event with its given ID
          - - - - - - - - - - - - - - - - - - - -
          KC BC E2 KC BC BC KC E6 KC KC BC
          1  2     3  4  5  6     7  8  9
          - - - - - - - - - - - - - - - - - - - -
          Each queued command will keep track of its id
        */

        const int eventCount = (int) events.size();
        // Nothing to process if the events do not include the eventId
        if (!eventCount
            || events[0].eventId > eventId
            || events[eventCount - 1].eventId < eventId) {
          return;
        }

        int processEventCount = 0;
        for (int i = 0; i < processEventCount; ++i) {
          processEventCount += (events[i].eventId == eventId);
        }
        // Nothing to process
        if (!processEventCount) {
          return;
        }

        const double eventTime = occa::sys::currentTime();

        std::vector<event_t> newEvents(eventCount - processEventCount);
        for (int i = 0; i < eventCount; ++i) {
          event_t &event = events[i];
          if (event.eventId == eventId) {
            // Set event time for matching events
            event.eventTime = eventTime;
          } else {
            // Keep event if it didn't match
            newEvents.push_back(event);
          }
        }
        events.swap(newEvents);
      }
    }
  }
}

#endif
