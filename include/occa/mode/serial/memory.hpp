#ifndef OCCA_MODES_SERIAL_MEMORY_HEADER
#define OCCA_MODES_SERIAL_MEMORY_HEADER

#include <occa/defines.hpp>
#include <occa/core/memory.hpp>

namespace occa {
  namespace serial {
    class memory : public occa::modeMemory_t {
    public:
      memory(modeDevice_t *modeDevice_,
             udim_t size_,
             const occa::properties &properties_ = occa::properties());
      ~memory();

      kernelArg makeKernelArg() const;

      modeMemory_t* addOffset(const dim_t offset);

      void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset,
                  const occa::properties &props) const;

      void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset,
                    const occa::properties &props);

      void copyFrom(const modeMemory_t *src,
                    const udim_t bytes,
                    const udim_t destOffset,
                    const udim_t srcOffset,
                    const occa::properties &props);
      void detach();
    };
  }
}

#endif
