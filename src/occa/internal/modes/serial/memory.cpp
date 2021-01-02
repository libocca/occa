#include <cstring>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/core/device.hpp>

namespace occa {
  namespace serial {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
      occa::modeMemory_t(modeDevice_, size_, properties_) {}

    memory::~memory() {
      if (ptr && isOrigin) {
        sys::free(ptr);
      }
      ptr = NULL;
      size = 0;
    }

    void* memory::getKernelArgPtr() const {
      return ptr;
    }

    modeMemory_t* memory::addOffset(const dim_t offset) {
      memory *m = new memory(modeDevice,
                             size - offset,
                             properties);
      m->ptr = ptr + offset;
      return m;
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset,
                        const occa::json &props) const {
      const void *srcPtr = ptr + offset;

      ::memcpy(dest, srcPtr, bytes);
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::json &props) {

      void *destPtr      = ptr + offset;
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

    void memory::detach() {
      ptr = NULL;
      size = 0;
    }
  }
}
