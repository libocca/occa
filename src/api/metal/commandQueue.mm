#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/api/metal/commandQueue.hpp>

namespace occa {
  namespace api {
    namespace metal {
      commandQueue_t::commandQueue_t(void *commandQueueObj_) :
        commandQueueObj(commandQueueObj_),
        lastCommandBufferObj(NULL) {}

      commandQueue_t::commandQueue_t(const commandQueue_t &other) :
        commandQueueObj(other.commandQueueObj),
        lastCommandBufferObj(other.lastCommandBufferObj) {}

      void commandQueue_t::free() {
          // Remove reference count
        if (commandQueueObj) {
          id<MTLCommandQueue> metalCommandQueue = (__bridge id<MTLCommandQueue>) commandQueueObj;
          metalCommandQueue = nil;
          commandQueueObj = NULL;
        }
        freeLastCommandBuffer();
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

      void commandQueue_t::setLastCommandBuffer(void *commandBufferObj) {
        if (commandBufferObj) {
          freeLastCommandBuffer();
          lastCommandBufferObj = commandBufferObj;
        }
      }
    }
  }
}

#endif
