#ifndef OCCA_INTERNAL_MODES_SERIAL_MEMORY_HEADER
#define OCCA_INTERNAL_MODES_SERIAL_MEMORY_HEADER

#include <occa/defines.hpp>
#include <occa/internal/core/memory.hpp>

namespace occa {
  namespace serial {
    class memory : public occa::modeMemory_t {
    public:
      memory(modeDevice_t *modeDevice_,
             udim_t size_,
             const occa::json &properties_ = occa::json());
      ~memory();

      void* getKernelArgPtr() const;

      modeMemory_t* addOffset(const dim_t offset);

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
      void detach();
    };
  }
}

#endif
