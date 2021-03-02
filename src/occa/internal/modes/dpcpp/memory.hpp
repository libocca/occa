#ifndef OCCA_MODES_DPCPP_MEMORY_HEADER
#define OCCA_MODES_DPCPP_MEMORY_HEADER

#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/dpcpp/polyfill.hpp>
// #include <occa/internal/modes/dpcpp/device.hpp>

namespace occa
{
  namespace dpcpp
  {
    class device;
    
    class memory : public occa::modeMemory_t
    {
      friend class dpcpp::device;

      friend void *getMappedPtr(occa::memory memory);

      friend occa::memory wrapMemory(occa::device device,
                                     void *dpcppMem,
                                     const udim_t bytes,
                                     const occa::json &props);

    public:
      memory(modeDevice_t *modeDevice_,
             udim_t size_,
             const occa::json &properties_ = occa::json());

      ~memory();

      void *getKernelArgPtr() const;

      modeMemory_t *addOffset(const dim_t offset);

      // void *getPtr(const occa::json &props);

      void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset = 0,
                  const occa::json &props = occa::json()) const;

      void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset = 0,
                    const occa::json &props = occa::json());

      void copyFrom(const modeMemory_t *src,
                    const udim_t bytes,
                    const udim_t destOffset = 0,
                    const udim_t srcOffset = 0,
                    const occa::json &props = occa::json());
      void detach();
    };
  } // namespace dpcpp
} // namespace occa

#endif
