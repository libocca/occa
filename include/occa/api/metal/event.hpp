#ifndef OCCA_INTERNAL_API_METAL_EVENT_HEADER
#define OCCA_INTERNAL_API_METAL_EVENT_HEADER

namespace occa {
  namespace api {
    namespace metal {
      class commandQueue_t;

      class event_t {
       public:
        commandQueue_t *commandQueue;
        void *eventObj;
        int eventId;
        void *commandBufferObj;
        double eventTime;

        event_t();

        event_t(commandQueue_t *commandQueue,
                void *eventObj_,
                const int eventId_,
                void *commandBufferObj_);

        event_t(const event_t &other);

        event_t& operator = (const event_t &other);

        void free();
        void freeCommandBuffer();

        void waitUntilCompleted();

        void setTime(const double eventTime_);
        double getTime() const;
      };
    }
  }
}

#endif
