#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#include <occa/modes/metal/memory.hpp>
#include <occa/modes/metal/device.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  namespace metal {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::properties &properties_) :
      occa::modeMemory_t(modeDevice_, size_, properties_) {}

    memory::~memory() {
      if (size) {
        // // Free mapped-host pointer
        // OCCA_OPENCL_ERROR("Mapped Free: clReleaseMemObject",
        //                   clReleaseMemObject(clMem));
        size = 0;
      }
    }

    kernelArg memory::makeKernelArg() const {
      kernelArgData arg;

      arg.modeMemory = const_cast<memory*>(this);
      arg.data.void_ = (void*) &metalBuffer;
      arg.size       = sizeof(void*);
      arg.info       = kArgInfo::usePointer;

      return kernelArg(arg);
    }

    modeMemory_t* memory::addOffset(const dim_t offset) {
      return NULL;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props) {
      const bool async = props.get("async", false);
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::properties &props) {
      const bool async = props.get("async", false);
    }

    void* memory::getPtr(const occa::properties &props) {
      return ptr;
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset,
                        const occa::properties &props) const {

      const bool async = props.get("async", false);
    }

    void memory::detach() {
      size = 0;
    }
  }
}

#endif
