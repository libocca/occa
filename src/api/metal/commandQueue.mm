#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/internal/api/metal/buffer.hpp>
#include <occa/internal/api/metal/commandQueue.hpp>
#include <occa/internal/api/metal/device.hpp>
#include <occa/internal/api/metal/event.hpp>
#include <occa/internal/utils/sys.hpp>

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

      commandQueue_t& commandQueue_t::operator = (const commandQueue_t &other) {
        device = other.device;
        commandQueueObj = other.commandQueueObj;
        lastCommandBufferObj = other.lastCommandBufferObj;
        events = other.events;
        lastCommandId = other.lastCommandId;
        return *this;
      }

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

        return event_t(const_cast<commandQueue_t*>(this),
                       eventObj,
                       lastCommandId,
                       lastCommandBufferObj);
      }

      void commandQueue_t::clearCommandBuffer(void *commandBufferObj) {
        if (commandBufferObj == lastCommandBufferObj) {
          freeLastCommandBuffer();
        }
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

      void commandQueue_t::finish() {
        if (lastCommandBufferObj) {
          id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) device->deviceObj;
          id<MTLCommandBuffer> metalCommandBuffer = (
            (__bridge id<MTLCommandBuffer>) lastCommandBufferObj
          );
          [metalCommandBuffer waitUntilCompleted];
          freeLastCommandBuffer();
        }
      }

      void commandQueue_t::memcpy(buffer_t &dest,
                                  const udim_t destOffset,
                                  const buffer_t &src,
                                  const udim_t srcOffset,
                                  const udim_t bytes,
                                  const bool async) {
        id<MTLCommandQueue> metalCommandQueue = (__bridge id<MTLCommandQueue>) commandQueueObj;
        id<MTLBuffer> srcMetalBuffer = (__bridge id<MTLBuffer>) src.bufferObj;
        id<MTLBuffer> destMetalBuffer = (__bridge id<MTLBuffer>) dest.bufferObj;

        // Initialize Metal command
        id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
        OCCA_ERROR("Command Queue: Create command buffer",
                   commandBuffer != nil);

        // The commandBuffer callback has to be set before commit is called on it
        setLastCommandBuffer((__bridge void*) commandBuffer);

        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        OCCA_ERROR("Command Queue: Create Blit encoder",
                   commandBuffer != nil);

        [blitEncoder copyFromBuffer:srcMetalBuffer
                       sourceOffset:srcOffset
                           toBuffer:destMetalBuffer
                  destinationOffset:destOffset
                               size:bytes];

        // Finish encoding and start the data transfer
        [blitEncoder endEncoding];
        [commandBuffer commit];

        if (!async) {
          finish();
        }
      }

      void commandQueue_t::memcpy(void *dest,
                                  const buffer_t &src,
                                  const udim_t srcOffset,
                                  const udim_t bytes,
                                  const bool async) {
        id<MTLCommandQueue> metalCommandQueue = (__bridge id<MTLCommandQueue>) commandQueueObj;
        id<MTLBuffer> srcMetalBuffer = (__bridge id<MTLBuffer>) src.bufferObj;

        // Initialize Metal command
        id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
        OCCA_ERROR("Command Queue: Create command buffer",
                   commandBuffer != nil);

        // The commandBuffer callback has to be set before commit is called on it
        setLastCommandBuffer((__bridge void*) commandBuffer);

        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        OCCA_ERROR("Command Queue: Create Blit encoder",
                   commandBuffer != nil);

        [blitEncoder synchronizeResource:srcMetalBuffer];

        // Finish encoding and start the data transfer to the CPU
        [blitEncoder endEncoding];
        [commandBuffer commit];

        // Make sure all edits in the GPU finish before using its data
        finish();

        ::memcpy(dest,
                 (void*) (((char*) src.getPtr()) + srcOffset),
                 bytes);
      }

      void commandQueue_t::memcpy(buffer_t &dest,
                                  const udim_t destOffset,
                                  const void *src,
                                  const udim_t bytes,
                                  const bool async) {
        id<MTLBuffer> destMetalBuffer = (__bridge id<MTLBuffer>) dest.bufferObj;

        // Make sure all uses of the GPU finish before updating its data
        finish();

        ::memcpy((void*) (((char*) dest.getPtr()) + destOffset),
                 src,
                 bytes);

        [destMetalBuffer didModifyRange:NSMakeRange(destOffset, bytes)];
      }
    }
  }
}

#endif
