#ifndef OCCA_API_METAL_DEVICE_HEADER
#define OCCA_API_METAL_DEVICE_HEADER

#include <occa/api/metal/buffer.hpp>
#include <occa/api/metal/commandQueue.hpp>
#include <occa/api/metal/event.hpp>
#include <occa/api/metal/function.hpp>

namespace occa {
  namespace io {
    class lock_t;
  }

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
                               const std::string &kernelName,
                               io::lock_t &lock) const;

        buffer_t malloc(const udim_t bytes,
                        const void *src) const;
      };
    }
  }
}

#endif
