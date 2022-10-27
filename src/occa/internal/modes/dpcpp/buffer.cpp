
#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/buffer.hpp>
#include <occa/internal/modes/dpcpp/memory.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace dpcpp {
    buffer::buffer(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
      occa::modeBuffer_t(modeDevice_, size_, properties_) {}

    buffer::~buffer() {
      if (!isWrapped && ptr) {
        auto& dpcpp_device = getDpcppDevice(modeDevice);
        OCCA_DPCPP_ERROR("Memory: Freeing SYCL alloc'd memory",
                         ::sycl::free(ptr,dpcpp_device.dpcppContext));
      }
      ptr = nullptr;
      size = 0;
    }

    void buffer::malloc(udim_t bytes) {

      dpcpp::device *device = reinterpret_cast<dpcpp::device*>(modeDevice);

      if (properties.get("host", false)) {
        ptr = static_cast<char *>(::sycl::malloc_host(bytes,
                                                      device->dpcppContext));
        OCCA_ERROR("DPCPP: malloc_host failed!", nullptr != ptr);
      } else if (properties.get("unified", false)) {
        ptr = static_cast<char *>(::sycl::malloc_shared(bytes,
                                                        device->dpcppDevice,
                                                        device->dpcppContext));
        OCCA_ERROR("DPCPP: malloc_shared failed!", nullptr != ptr);
      } else {
        ptr = static_cast<char *>(::sycl::malloc_device(bytes,
                                                        device->dpcppDevice,
                                                        device->dpcppContext));
        OCCA_ERROR("DPCPP: malloc_device failed!", nullptr != ptr);
      }
      size = bytes;
    }

    void buffer::wrapMemory(const void *ptr_,
                            const udim_t bytes) {

      ptr = (char*) const_cast<void*>(ptr_);
      size = bytes;
      isWrapped = true;
    }

    modeMemory_t* buffer::slice(const dim_t offset,
                                const udim_t bytes) {
      return new dpcpp::memory(this, bytes, offset);
    }

    void buffer::detach() {
      ptr = nullptr;
      size = 0;
      isWrapped = false;
    }
  } // namespace dpcpp
} // namespace occa
