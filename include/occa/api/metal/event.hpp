#ifndef OCCA_API_METAL_EVENT_HEADER
#define OCCA_API_METAL_EVENT_HEADER

namespace occa {
  namespace api {
    namespace metal {
      class event_t {
       public:
        void *eventObj;
        int eventId;
        void *commandBufferObj;
        double eventTime;

        event_t();

        event_t(void *eventObj_,
                const int eventId_,
                void *commandBufferObj_);

        event_t(const event_t &other);

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
