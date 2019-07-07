#ifndef OCCA_API_METAL_KERNEL_HEADER
#define OCCA_API_METAL_KERNEL_HEADER

#include <occa/types.hpp>

namespace occa {
  class kernelArgData;

  namespace api {
    namespace metal {
      class commandQueue_t;
      class device_t;

      class kernel_t {
       public:
        device_t *device;
        void *functionObj;
        void *pipelineStateObj;

        kernel_t();

        kernel_t(device_t *device_,
                 void *functionObj_);

        kernel_t(const kernel_t &other);

        void free();

        void run(commandQueue_t &commandQueue,
                 occa::dim outerDims,
                 occa::dim innerDims,
                 const std::vector<kernelArgData> &arguments);
      };
    }
  }
}

#endif
