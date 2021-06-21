#ifndef OCCA_INTERNAL_API_METAL_DEVICE_HEADER
#define OCCA_INTERNAL_API_METAL_DEVICE_HEADER

#include <occa/internal/api/metal/buffer.hpp>
#include <occa/internal/api/metal/commandQueue.hpp>
#include <occa/internal/api/metal/event.hpp>
#include <occa/internal/api/metal/function.hpp>

namespace occa {
  namespace api {
    namespace metal {
      class device_t {
       public:
        void *deviceObj;

        device_t(void *deviceObj_ = NULL);
        device_t(const device_t &other);

        device_t& operator = (const device_t &other);

        void free();

        std::string getName() const;
        udim_t getMemorySize() const;

        dim getMaxOuterDims() const;
        dim getMaxInnerDims() const;

        commandQueue_t createCommandQueue() const;

        function_t buildKernel(const std::string &metallibFilename,
                               const std::string &kernelName) const;

        buffer_t malloc(const udim_t bytes,
                        const void *src) const;
      };
    }
  }
}

#endif
