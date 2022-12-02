#ifndef OCCA_MODES_DPCPP_MEMORY_HEADER
#define OCCA_MODES_DPCPP_MEMORY_HEADER

#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/dpcpp/polyfill.hpp>
#include <occa/internal/modes/dpcpp/buffer.hpp>
#include <occa/internal/modes/dpcpp/memoryPool.hpp>

namespace occa
{
  namespace dpcpp
  {
    class device;
    
    class memory : public occa::modeMemory_t
    {
    public:
      memory(buffer *b,
             udim_t size_, dim_t offset_);
      memory(memoryPool *memPool,
             udim_t size_, dim_t offset_);

      virtual ~memory();

      void *getKernelArgPtr() const override;

      void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset = 0,
                  const occa::json &props = occa::json()) const override;

      void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset = 0,
                    const occa::json &props = occa::json()) override;

      void copyFrom(const modeMemory_t *src,
                    const udim_t bytes,
                    const udim_t destOffset = 0,
                    const udim_t srcOffset = 0,
                    const occa::json &props = occa::json()) override;

      void* unwrap() override;
    };
  } // namespace dpcpp
} // namespace occa

#endif
