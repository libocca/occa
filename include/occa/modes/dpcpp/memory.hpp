#ifndef OCCA_MODES_DPCPP_MEMORY_HEADER
#define OCCA_MODES_DPCPP_MEMORY_HEADER

#include <occa/core/memory.hpp>
#include <occa/modes/dpcpp/polyfill.hpp>
#include <occa/modes/dpcpp/device.hpp>

namespace occa {
  namespace sycl {
    class memory : public occa::modeMemory_t {
      friend class dpcpp::device;
/* check these friend functions*/
      friend cl_mem getCLMemory(occa::memory memory);

      friend void* getMappedPtr(occa::memory memory);

      friend occa::memory wrapMemory(occa::device device,
                                     void *dpcppMem,
                                     const udim_t bytes,
                                     const occa::properties &props);

    private:
      void *rootDpcppMem;
      dim_t rootOffset;

      void *dpcppMem;
      void *mappedPtr;
    public:
      memory(modeDevice_t *modeDevice_,
             udim_t size_,
             const occa::properties &properties_ = occa::properties());
      ~memory();

      kernelArg makeKernelArg() const;
      ::sycl::queue *getCommandQueue() const;
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

#endif
