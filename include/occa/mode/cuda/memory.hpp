#include <occa/defines.hpp>

#if OCCA_CUDA_ENABLED
#  ifndef OCCA_MODES_CUDA_MEMORY_HEADER
#  define OCCA_MODES_CUDA_MEMORY_HEADER

#include <cuda.h>

#include <occa/core/memory.hpp>

namespace occa {
  namespace cuda {
    class device;

    class memory : public occa::modeMemory_t {
      friend class cuda::device;

      friend occa::memory wrapMemory(occa::device device,
                                     void *ptr,
                                     const udim_t bytes,
                                     const occa::properties &props);

      friend void* getMappedPtr(occa::memory mem);

    public:
      CUdeviceptr &cuPtr;
      char *mappedPtr;
      bool isUnified;

      memory(modeDevice_t *modeDevice_,
             udim_t size_,
             const occa::properties &properties_ = occa::properties());
      ~memory();

      CUstream& getCuStream() const;

      kernelArg makeKernelArg() const;

      modeMemory_t* addOffset(const dim_t offset);

      void* getPtr(const occa::properties &props);

      void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset = 0,
                  const occa::properties &props = occa::properties()) const;

      void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset = 0,
                    const occa::properties &props = occa::properties());

      void copyFrom(const modeMemory_t *src,
                    const udim_t bytes,
                    const udim_t destOffset = 0,
                    const udim_t srcOffset = 0,
                    const occa::properties &props = occa::properties());
      void detach();
    };
  }
}

#  endif
#endif
