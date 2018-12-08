#include <occa/defines.hpp>

#if OCCA_HIP_ENABLED

#include <occa/mode/hip/memory.hpp>
#include <occa/mode/hip/device.hpp>
#include <occa/mode/hip/utils.hpp>

namespace occa {
  namespace hip {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::properties &properties_) :
      occa::modeMemory_t(modeDevice_, size_, properties_),
      hipPtr((hipDeviceptr_t&) ptr),
      mappedPtr(NULL) {}

    memory::~memory() {
      if (!isOrigin) {
        hipPtr = 0;
        mappedPtr = NULL;
        size = 0;
        return;
      }

      if (mappedPtr) {
        OCCA_HIP_ERROR("Device: mappedFree()",
                       hipHostFree(mappedPtr));
        mappedPtr = NULL;
      } else if (hipPtr) {
        hipHostFree(hipPtr);
        hipPtr = 0;
      }
      size = 0;
    }

    kernelArg memory::makeKernelArg() const {
      kernelArgData arg;

      arg.modeMemory = const_cast<memory*>(this);
      arg.data.void_ = (void*) hipPtr;
      arg.size       = sizeof(void*);
      arg.info       = kArgInfo::usePointer;

      return kernelArg(arg);
    }

    modeMemory_t* memory::addOffset(const dim_t offset) {
      memory *m = new memory(modeDevice,
                             size - offset,
                             properties);
      m->hipPtr = (char*) hipPtr + offset;
      if (mappedPtr) {
        m->mappedPtr = mappedPtr + offset;
      }
      return m;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props) {
      const hipStream_t &stream = *((hipStream_t*) modeDevice->currentStream);
      const bool async = props.get("async", false);

      if (!async) {
        OCCA_HIP_ERROR("Memory: Copy From",
                       hipMemcpyHtoD((char*) hipPtr + offset,
                                     const_cast<void*>(src),
                                     bytes) );
      } else {
        OCCA_HIP_ERROR("Memory: Async Copy From",
                       hipMemcpyHtoDAsync((char*) hipPtr + offset,
                                          const_cast<void*>(src),
                                          bytes,
                                          stream) );
      }
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::properties &props) {
      const hipStream_t &stream = *((hipStream_t*) modeDevice->currentStream);
      const bool async = props.get("async", false);

      if (!async) {
        OCCA_HIP_ERROR("Memory: Copy From",
                       hipMemcpyDtoD((char*) hipPtr + destOffset,
                                     (char*) ((memory*) src)->hipPtr + srcOffset,
                                     bytes) );
      } else {
        OCCA_HIP_ERROR("Memory: Async Copy From",
                       hipMemcpyDtoDAsync((char*) hipPtr + destOffset,
                                          (char*) ((memory*) src)->hipPtr + srcOffset,
                                          bytes,
                                          stream) );
      }
    }

    void* memory::getPtr(const occa::properties &props) {
      if (props.get("mapped", false)) {
        return mappedPtr;
      }
      return ptr;
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset,
                        const occa::properties &props) const {
      const hipStream_t &stream = *((hipStream_t*) modeDevice->currentStream);
      const bool async = props.get("async", false);

      if (!async) {
        OCCA_HIP_ERROR("Memory: Copy From",
                       hipMemcpyDtoH(dest,
                                     (char*) hipPtr + offset,
                                     bytes) );
      } else {
        OCCA_HIP_ERROR("Memory: Async Copy From",
                       hipMemcpyDtoHAsync(dest,
                                          (char*) hipPtr + offset,
                                          bytes,
                                          stream) );
      }
    }

    void memory::detach() {
      hipPtr = 0;
      size = 0;
    }
  }
}

#endif
