#ifndef OCCA_INTERNAL_CORE_MEMORY_HEADER
#define OCCA_INTERNAL_CORE_MEMORY_HEADER

#include <occa/core/memory.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/gc.hpp>

namespace occa {
  class kernelArgData;
  class modeBuffer_t;

  class modeMemory_t : public gc::ringEntry_t {
   public:
    gc::ring_t<memory> memoryRing;
    occa::modeBuffer_t *modeBuffer;

    char *ptr;

    const dtype_t *dtype_;
    udim_t size;
    dim_t offset;

    modeMemory_t(modeBuffer_t *modeBuffer_,
                 udim_t size_, dim_t offset_);

    void dontUseRefs();
    void addMemoryRef(memory *mem);
    void removeMemoryRef(memory *mem);
    void removeModeMemoryRef();
    bool needsFree() const;


    modeDevice_t* getModeDevice() const;
    const occa::json& properties() const;

    modeMemory_t* slice(const dim_t offset_,
                        const udim_t bytes);

    void detach();

    //---[ Virtual Methods ]------------
    virtual ~modeMemory_t();

    virtual void* getKernelArgPtr() const = 0;

    virtual void* getPtr() const;


    virtual void copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset = 0,
                        const occa::json &props = occa::json()) const = 0;

    virtual void copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset = 0,
                          const occa::json &props = occa::json()) = 0;

    virtual void copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset = 0,
                          const udim_t srcOffset = 0,
                          const occa::json &props = occa::json()) = 0;

    virtual void* unwrap() = 0;
    //==================================

    //---[ Friend Functions ]-----------
    friend void memcpy(void *dest, void *src,
                       const dim_t bytes,
                       const occa::json &props);
  };
}

#endif
