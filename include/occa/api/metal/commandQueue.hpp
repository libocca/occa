#ifndef OCCA_API_METAL_COMMANDQUEUE_HEADER
#define OCCA_API_METAL_COMMANDQUEUE_HEADER

namespace occa {
  namespace api {
    namespace metal {
      class commandQueue_t {
       public:
        void *commandQueueObj;

        commandQueue_t(void *commandQueueObj_ = NULL);
        commandQueue_t(const commandQueue_t &other);

        void free();
      };
    }
  }
}

#endif
