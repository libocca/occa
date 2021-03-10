#ifndef OCCA_INTERNAL_API_METAL_COMMANDQUEUE_HEADER
#define OCCA_INTERNAL_API_METAL_COMMANDQUEUE_HEADER

#include <cstddef>
#include <vector>

#include <occa/internal/api/metal/event.hpp>
#include <occa/types.hpp>

namespace occa {
  namespace api {
    namespace metal {
      class buffer_t;
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

        commandQueue_t& operator = (const commandQueue_t &other);

        void free();
        void freeLastCommandBuffer();

        event_t createEvent() const;

        void clearCommandBuffer(void *commandBufferObj);
        void setLastCommandBuffer(void *commandBufferObj);

        void processEvents(const int eventId);

        void finish();

        void memcpy(buffer_t &dest,
                    const udim_t destOffset,
                    const buffer_t &src,
                    const udim_t srcOffset,
                    const udim_t bytes,
                    const bool async);

        void memcpy(void *dest,
                    const buffer_t &src,
                    const udim_t srcOffset,
                    const udim_t bytes,
                    const bool async);

        void memcpy(buffer_t &dest,
                    const udim_t destOffset,
                    const void *src,
                    const udim_t bytes,
                    const bool async);
      };
    }
  }
}

#endif
