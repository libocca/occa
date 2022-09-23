#ifndef OCCA_INTERNAL_CORE_MEMORYPOOL_HEADER
#define OCCA_INTERNAL_CORE_MEMORYPOOL_HEADER

#include <occa/core/memoryPool.hpp>
#include <occa/internal/core/buffer.hpp>
#include <set>

namespace occa {
  using experimental::memoryPool;

  class modeMemoryPool_t : public modeBuffer_t {
   public:
    struct compare {
      bool operator()(const modeMemory_t* a, const modeMemory_t* b) const {
        return (a->offset  < b->offset) ||
               (a->offset == b->offset &&  a->size < b->size);
      };
    };
    typedef std::set<modeMemory_t*, compare> reservationSet;

    gc::ring_t<memoryPool> memoryPoolRing;

    reservationSet reservations;

    udim_t alignment;
    udim_t reserved;

    modeBuffer_t* buffer;

    bool verbose;

    modeMemoryPool_t(modeDevice_t *modeDevice_,
                     const occa::json &json_);
    ~modeMemoryPool_t();

    udim_t numReservations() const;

    bool deleteOnFree() override {return false;}

    modeMemory_t* reserve(const udim_t bytes);

    void resize(const udim_t bytes);

    void setAlignment(const udim_t newAlignment);

    void dontUseRefs() override;
    bool needsFree() const override;
    void addMemoryPoolRef(memoryPool *memPool);
    void removeMemoryPoolRef(memoryPool *memPool);
    void addModeMemoryRef(modeMemory_t *mem) override;
    void removeModeMemoryRef(modeMemory_t *mem) override;

   private:
    virtual modeBuffer_t* makeBuffer()=0;
    virtual modeMemory_t* slice(const dim_t offset, const udim_t bytes) = 0;
    virtual void setPtr(modeMemory_t* mem, modeBuffer_t* buf, const dim_t offset)=0;
    virtual void memcpy(modeBuffer_t* dst, const dim_t dstOffset,
                        modeBuffer_t* src, const dim_t srcOffset,
                        const udim_t bytes) = 0;
  };
}

#endif
