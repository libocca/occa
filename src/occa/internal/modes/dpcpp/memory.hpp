#ifndef OCCA_MODES_DPCPP_MEMORY_HEADER
#define OCCA_MODES_DPCPP_MEMORY_HEADER

#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/dpcpp/polyfill.hpp>

namespace occa
{
  namespace dpcpp
  {
    class device;
    
    class memory : public occa::modeMemory_t
    {
    public:
      memory(modeBuffer_t *modeBuffer_,
             udim_t size_, dim_t offset_);

      virtual ~memory();

      virtual void *getKernelArgPtr() const override;

      virtual void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset = 0,
                  const occa::json &props = occa::json()) const override;

      virtual void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset = 0,
                    const occa::json &props = occa::json()) override;

      virtual void copyFrom(const modeMemory_t *src,
                    const udim_t bytes,
                    const udim_t destOffset = 0,
                    const udim_t srcOffset = 0,
                    const occa::json &props = occa::json()) override;
    };
  } // namespace dpcpp
} // namespace occa

#endif
