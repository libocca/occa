#include <occa/mode/serial/memory.hpp>
#include <occa/tools/sys.hpp>
#include <occa/core/device.hpp>

namespace occa {
  namespace serial {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::properties &properties_) :
      occa::modeMemory_t(modeDevice_, size_, properties_) {}

    memory::~memory() {
      if (ptr) {
        if (isOrigin) {
          sys::free(ptr);
        }
        ptr = NULL;
        size = 0;
      }
    }

    kernelArg memory::makeKernelArg() const {
      kernelArgData arg;

      arg.modeMemory = const_cast<memory*>(this);
      arg.data.void_ = ptr;
      arg.size       = sizeof(void*);
      arg.info       = kArgInfo::usePointer;

      return kernelArg(arg);
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
                        const occa::properties &props) const {
      const void *srcPtr = ptr + offset;

      ::memcpy(dest, srcPtr, bytes);
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props) {

      void *destPtr      = ptr + offset;
      const void *srcPtr = src;

      ::memcpy(destPtr, srcPtr, bytes);
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::properties &props) {

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
