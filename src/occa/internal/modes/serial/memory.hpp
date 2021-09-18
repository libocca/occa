#ifndef OCCA_INTERNAL_MODES_SERIAL_MEMORY_HEADER
#define OCCA_INTERNAL_MODES_SERIAL_MEMORY_HEADER

#include <occa/defines.hpp>
#include <occa/internal/core/memory.hpp>

namespace occa {
  namespace serial {
    class memory : public occa::modeMemory_t {
    public:
      memory(modeBuffer_t *modeBuffer_,
             udim_t size_, dim_t offset_);
      ~memory();

      void* getKernelArgPtr() const;

      void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset,
                  const occa::json &props) const;

      void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset,
                    const occa::json &props);

      void copyFrom(const modeMemory_t *src,
                    const udim_t bytes,
                    const udim_t destOffset,
                    const udim_t srcOffset,
                    const occa::json &props);
    };
  }
}

#endif
