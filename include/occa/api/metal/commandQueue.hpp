#ifndef OCCA_API_METAL_COMMANDQUEUE_HEADER
#define OCCA_API_METAL_COMMANDQUEUE_HEADER

#include <cstddef>
#include <vector>

#include <occa/api/metal/event.hpp>

namespace occa {
  namespace api {
    namespace metal {
      class device_t;

      class commandQueue_t {
       public:
        device_t *device;
        void *commandQueueObj;
        void *lastCommandBufferObj;
        int lastCommandId;
        std::vector<event_t> events;

        commandQueue_t();

        commandQueue_t(device_t *device_,
                       void *commandQueueObj_);

        commandQueue_t(const commandQueue_t &other);

        void free();
        void freeLastCommandBuffer();

        event_t createEvent() const;

        void clearCommandBuffer(void *commandBufferObj);
        void setLastCommandBuffer(void *commandBufferObj);

        void processEvents(const int eventId);

        void finish();
      };
    }
  }
}

#endif
