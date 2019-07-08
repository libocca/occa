#ifndef OCCA_MODES_METAL_MEMORY_HEADER
#define OCCA_MODES_METAL_MEMORY_HEADER

#include <occa/core/memory.hpp>
#include <occa/api/metal.hpp>

namespace occa {
  namespace metal {
    class device;

    class memory : public occa::modeMemory_t {
      friend class metal::device;

    private:
      api::metal::buffer_t metalBuffer;
      udim_t bufferOffset;

    public:
      memory(modeDevice_t *modeDevice_,
             udim_t size_,
             const occa::properties &properties_ = occa::properties());
      ~memory();

      kernelArg makeKernelArg() const;

      modeMemory_t* addOffset(const dim_t offset);

      const api::metal::buffer_t& getMetalBuffer();

      void* getPtr(const occa::properties &props);

      udim_t getOffset() const;

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
