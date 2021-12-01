#ifndef OCCA_INTERNAL_CORE_MEMORYPOOL_HEADER
#define OCCA_INTERNAL_CORE_MEMORYPOOL_HEADER

#include <occa/core/memoryPool.hpp>
#include <occa/internal/core/buffer.hpp>
#include <set>

namespace occa {
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

    udim_t reserved;

    modeMemoryPool_t(modeDevice_t *modeDevice_,
                     const occa::json &json_);

    bool deleteOnFree() {return false;}

    modeMemory_t* reserve(const udim_t bytes);

    void dontUseRefs() override;
    bool needsFree() const override;
    void addMemoryPoolRef(memoryPool *memPool);
    void removeMemoryPoolRef(memoryPool *memPool);
    void addModeMemoryRef(modeMemory_t *mem) override;
    void removeModeMemoryRef(modeMemory_t *mem) override;

    //---[ Virtual Methods ]------------
    virtual ~modeMemoryPool_t();

    virtual void resize(const udim_t bytes)=0;
  };
}

#endif
