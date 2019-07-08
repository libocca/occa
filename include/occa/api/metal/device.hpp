#ifndef OCCA_API_METAL_DEVICE_HEADER
#define OCCA_API_METAL_DEVICE_HEADER

#include <occa/types.hpp>
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
        void *libraryObj;

        device_t(void *deviceObj_ = NULL);
        device_t(const device_t &other);

        void free();

        std::string getName() const;
        udim_t getMemorySize() const;

        dim getMaxOuterDims() const;
        dim getMaxInnerDims() const;

        commandQueue_t createCommandQueue() const;

        function_t buildKernel(const std::string &source,
                               const std::string &kernelName,
                               io::lock_t &lock) const;

        buffer_t malloc(const udim_t bytes,
                        const void *src) const;

        void memcpy(buffer_t &dest,
                    const udim_t destOffset,
                    const buffer_t &src,
                    const udim_t srcOffset,
                    const udim_t bytes,
                    const bool async) const;

        void memcpy(void *dest,
                    const buffer_t &src,
                    const udim_t srcOffset,
                    const udim_t bytes,
                    const bool async) const;

        void memcpy(buffer_t &dest,
                    const udim_t destOffset,
                    const void *src,
                    const udim_t bytes,
                    const bool async) const;
      };
    }
  }
}

#endif
