#ifndef OCCA_INTERNAL_MODES_SERIAL_MEMORY_HEADER
#define OCCA_INTERNAL_MODES_SERIAL_MEMORY_HEADER

#include <occa/defines.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/serial/buffer.hpp>
#include <occa/internal/modes/serial/memoryPool.hpp>

namespace occa {
  namespace serial {
    class memory : public occa::modeMemory_t {
    public:
      memory(buffer *b,
             udim_t size_, dim_t offset_);
      memory(memoryPool *memPool,
             udim_t size_, dim_t offset_);
      virtual ~memory();

      void* getKernelArgPtr() const override;

      void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset,
                  const occa::json &props) const override;

      void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset,
                    const occa::json &props) override;

      void copyFrom(const modeMemory_t *src,
                    const udim_t bytes,
                    const udim_t destOffset,
                    const udim_t srcOffset,
                    const occa::json &props) override;

      void* unwrap() override;
    };
  }
}

#endif
