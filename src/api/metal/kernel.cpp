#include <occa/defines.hpp>

#if !OCCA_METAL_ENABLED

#include <occa/api/metal/kernel.hpp>

namespace occa {
  namespace api {
    namespace metal {
      kernel_t::kernel_t() {}

      kernel_t::kernel_t(const kernel_t &other) {}

      void kernel_t::clearArguments() {}

      void kernel_t::addArgument(const int index,
                                 const kernelArgData &arg) {}

      void kernel_t::run(occa::dim outerDims,
                         occa::dim innerDims) {}

      void kernel_t::free() {}
    }
  }
}

#endif
