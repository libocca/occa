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

        kernel_t();
        kernel_t(const kernel_t &other);

        void clearArguments();

        void addArgument(const int index,
                         const kernelArgData &arg);

        void run(occa::dim outerDims,
                 occa::dim innerDims);

        void free();
      };
    }
  }
}

#endif
