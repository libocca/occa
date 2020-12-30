#ifndef OCCA_INTERNAL_API_METAL_KERNEL_HEADER
#define OCCA_INTERNAL_API_METAL_KERNEL_HEADER

#include <occa/types.hpp>

namespace occa {
  class kernelArgData;

  namespace api {
    namespace metal {
      class commandQueue_t;
      class device_t;

      class function_t {
       public:
        device_t *device;
        void *libraryObj;
        void *functionObj;
        void *pipelineStateObj;

        function_t();

        function_t(device_t *device_,
                   void *libraryObj_,
                   void *functionObj_);

        function_t(const function_t &other);

        function_t& operator = (const function_t &other);

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
