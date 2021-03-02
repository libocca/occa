#ifndef OCCA_INTERNAL_MODES_OPENCL_MEMORY_HEADER
#define OCCA_INTERNAL_MODES_OPENCL_MEMORY_HEADER

#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/opencl/polyfill.hpp>

namespace occa {
  namespace opencl {
    class device;

    class memory : public occa::modeMemory_t {
      friend class opencl::device;

      friend cl_mem getCLMemory(occa::memory memory);

      friend occa::memory wrapMemory(occa::device device,
                                     cl_mem clMem,
                                     const udim_t bytes,
                                     const occa::json &props);

    private:
      cl_mem *rootClMem;
      dim_t rootOffset;

      cl_mem clMem;
      bool useHostPtr;

    public:
      memory(modeDevice_t *modeDevice_,
             udim_t size_,
             const occa::json &properties_ = occa::json());
      ~memory();

      cl_command_queue& getCommandQueue() const;

      void* getKernelArgPtr() const;

      modeMemory_t* addOffset(const dim_t offset);

      void* getPtr();

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
  }
}

#endif
