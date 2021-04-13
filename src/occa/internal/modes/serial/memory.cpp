#include <cstring>
#include <occa/internal/modes/serial/buffer.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/core/device.hpp>

namespace occa {
  namespace serial {
    memory::memory(modeBuffer_t *modeBuffer_,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(modeBuffer_, size_, offset_) {
      buffer *b = dynamic_cast<buffer*>(modeBuffer);
      ptr = b->ptr + offset;
    }

    memory::~memory() {}

    void* memory::getKernelArgPtr() const {
      return ptr;
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset_,
                        const occa::json &props) const {
      const void *srcPtr = ptr + offset_;

      ::memcpy(dest, srcPtr, bytes);
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset_,
                          const occa::json &props) {

      void *destPtr      = ptr + offset_;
      const void *srcPtr = src;

      ::memcpy(destPtr, srcPtr, bytes);
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::json &props) {

      void *destPtr      = ptr + destOffset;
      const void *srcPtr = src->ptr + srcOffset;

      ::memcpy(destPtr, srcPtr, bytes);
    }
  }
}
