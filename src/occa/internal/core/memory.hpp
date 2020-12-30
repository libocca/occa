#ifndef OCCA_INTERNAL_CORE_MEMORY_HEADER
#define OCCA_INTERNAL_CORE_MEMORY_HEADER

#include <occa/core/memory.hpp>
#include <occa/types/properties.hpp>
#include <occa/internal/utils/gc.hpp>

namespace occa {
  class modeMemory_t : public gc::ringEntry_t {
  public:
    int memInfo;
    occa::properties properties;

    gc::ring_t<memory> memoryRing;

    char *ptr;
    char *uvaPtr;

    occa::modeDevice_t *modeDevice;

    const dtype_t *dtype_;
    udim_t size;
    bool isOrigin;

    modeMemory_t(modeDevice_t *modeDevice_,
                 udim_t size_,
                 const occa::properties &properties_);

    void dontUseRefs();
    void addMemoryRef(memory *mem);
    void removeMemoryRef(memory *mem);
    bool needsFree() const;

    bool isManaged() const;
    bool inDevice() const;
    bool isStale() const;

    //---[ Virtual Methods ]------------
    virtual ~modeMemory_t() = 0;

    virtual kernelArg makeKernelArg() const = 0;

    virtual modeMemory_t* addOffset(const dim_t offset) = 0;

    virtual void* getPtr(const occa::properties &props);

    virtual void copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset = 0,
                        const occa::properties &props = occa::properties()) const = 0;

    virtual void copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset = 0,
                          const occa::properties &props = occa::properties()) = 0;

    virtual void copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset = 0,
                          const udim_t srcOffset = 0,
                          const occa::properties &props = occa::properties()) = 0;

    virtual void detach() = 0;
    //==================================

    //---[ Friend Functions ]-----------
    friend void memcpy(void *dest, void *src,
                       const dim_t bytes,
                       const occa::properties &props);

    friend void startManaging(void *ptr);
    friend void stopManaging(void *ptr);

    friend void syncToDevice(void *ptr, const dim_t bytes);
    friend void syncToHost(void *ptr, const dim_t bytes);

    friend void syncMemToDevice(occa::modeMemory_t *mem,
                                const dim_t bytes,
                                const dim_t offset);

    friend void syncMemToHost(occa::modeMemory_t *mem,
                              const dim_t bytes,
                              const dim_t offset);
  };
}

#endif
