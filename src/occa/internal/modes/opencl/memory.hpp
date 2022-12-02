#ifndef OCCA_INTERNAL_MODES_OPENCL_MEMORY_HEADER
#define OCCA_INTERNAL_MODES_OPENCL_MEMORY_HEADER

#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/opencl/polyfill.hpp>
#include <occa/internal/modes/opencl/buffer.hpp>
#include <occa/internal/modes/opencl/memoryPool.hpp>

namespace occa {
  namespace opencl {
    // class device;

    class memory : public occa::modeMemory_t {
      friend cl_mem getCLMemory(occa::memory memory);

    public:
      cl_mem clMem;
      bool useHostPtr;

    public:
      memory(buffer *b,
             udim_t size_, dim_t offset_);
      memory(memoryPool *memPool,
             udim_t size_, dim_t offset_);
      virtual ~memory();

      cl_command_queue& getCommandQueue() const;

      void* getKernelArgPtr() const override;

      void* getPtr() const override;

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
  }
}

#endif
