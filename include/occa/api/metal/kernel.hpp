#ifndef OCCA_API_METAL_KERNEL_HEADER
#define OCCA_API_METAL_KERNEL_HEADER

#include <occa/types.hpp>

namespace occa {
  class kernelArgData;

  namespace api {
    namespace metal {
      class kernel_t {
       public:
        void *obj;

        kernel_t(void *obj_ = NULL);
        kernel_t(const kernel_t &other);

        void free();

        void clearArguments();

        void addArgument(const int index,
                         const kernelArgData &arg);

        void run(occa::dim outerDims,
                 occa::dim innerDims);
      };
    }
  }
}

#endif
