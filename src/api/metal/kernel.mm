#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/api/metal/kernel.hpp>

namespace occa {
  namespace api {
    namespace metal {
      kernel_t::kernel_t(void *obj_) :
      obj(obj_) {}

      kernel_t::kernel_t(const kernel_t &other) :
      obj(other.obj) {}

      void kernel_t::free() {
        if (obj) {
          // Remove reference count
          id<MTLFunction> kernel = (__bridge id<MTLFunction>) obj;
          kernel = nil;
          obj = NULL;
        }
      }

      void kernel_t::clearArguments() {}

      void kernel_t::addArgument(const int index,
                                 const kernelArgData &arg) {}

      void kernel_t::run(occa::dim outerDims,
                         occa::dim innerDims) {}
    }
  }
}

#endif
