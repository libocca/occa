#include <occa/internal/modes/metal/device.hpp>
#include <occa/internal/modes/metal/buffer.hpp>
#include <occa/internal/modes/metal/memory.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace metal {
    buffer::buffer(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
      occa::modeBuffer_t(modeDevice_, size_, properties_) {}

    buffer::~buffer() {
      if (!isWrapped && metalBuffer.bufferObj) {
        metalBuffer.free();
      }
    }

    void buffer::malloc(udim_t bytes) {
      metalBuffer = dynamic_cast<metal::device*>(modeDevice)
                      ->metalDevice.malloc(bytes, NULL);
      size = bytes;
    }

    void buffer::wrapMemory(const void *ptr_,
                            const udim_t bytes) {
      metalBuffer = api::metal::buffer_t(const_cast<void*>(ptr_));
      size = bytes;
      isWrapped = true;
    }

    modeMemory_t* buffer::slice(const dim_t offset,
                                const udim_t bytes) {
      return new metal::memory(this, bytes, offset);
    }

    void buffer::detach() {
      metalBuffer.bufferObj = NULL;
      size = 0;
      isWrapped = false;
    }
  }
}
