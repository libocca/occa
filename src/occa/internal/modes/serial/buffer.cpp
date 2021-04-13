#include <cstring>
#include <occa/internal/modes/serial/buffer.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/core/device.hpp>

namespace occa {
  namespace serial {
    buffer::buffer(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
      occa::modeBuffer_t(modeDevice_, size_, properties_) {}

    buffer::~buffer() {

      if (!isWrapped && ptr) {
        if (properties.get("use_host_pointer", false)) {
          if (properties.get("own_host_pointer", false)) {
            sys::free(ptr);
          }
        } else {
          sys::free(ptr);
        }
      }
      ptr = NULL;
    }

    void buffer::malloc(udim_t bytes) {
      ptr = (char*) sys::malloc(bytes);
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
      return new serial::memory(this, bytes, offset);
    }

    void buffer::detach() {
      ptr = NULL;
      size = 0;
      isWrapped = false;
    }
  }
}
