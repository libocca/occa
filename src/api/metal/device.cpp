#include <occa/defines.hpp>

#if !OCCA_METAL_ENABLED

#include <occa/api/metal/device.hpp>

namespace occa {
  namespace api {
    namespace metal {
      device_t::device_t() {}

      device_t::device_t(const device_t &other) {}

      void device_t::free() {}

      std::string device_t::getName() const {
        return "";
      }

      udim_t device_t::getMemorySize() const {
        return 0;
      }

      dim device_t::getMaxOuterDims() const {
        return dim();
      }

      dim device_t::getMaxInnerDims() const {
        return dim();
      }

      commandQueue_t device_t::createCommandQueue() const {
        return commandQueue_t();
      }

      event_t device_t::createEvent() const {
        return event_t();
      }

      kernel_t device_t::buildKernel(const std::string &source,
                                     const std::string &kernelName,
                                     io::lock_t &lock) const {
        return kernel_t();
      }

      buffer_t device_t::malloc(const udim_t bytes,
                                const void *src) const {
        return buffer_t();
      }

      void device_t::memcpy(buffer_t &dest,
                            const udim_t destOffset,
                            const buffer_t &src,
                            const udim_t srcOffset,
                            const udim_t bytes,
                            const bool async) const {}

      void device_t::memcpy(void *dest,
                            const buffer_t &src,
                            const udim_t srcOffset,
                            const udim_t bytes,
                            const bool async) const {}

      void device_t::memcpy(buffer_t &dest,
                            const udim_t destOffset,
                            const void *src,
                            const udim_t bytes,
                            const bool async) const {}

      void device_t::waitFor(event_t &event) const {}

      void device_t::device_t::finish() const {}
    }
  }
}

#endif
