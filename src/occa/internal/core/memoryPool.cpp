#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/modes/serial/memoryPool.hpp>

namespace occa {
  modeMemoryPool_t::modeMemoryPool_t(modeDevice_t *modeDevice_,
                                     const occa::json &properties_) :
    modeBuffer_t(modeDevice_, 0, properties_),
    reserved(0) {

  }

  modeMemoryPool_t::~modeMemoryPool_t() {
    // NULL all wrappers
    while (memoryPoolRing.head) {
      memoryPool *memPool = (memoryPool*) memoryPoolRing.head;
      memoryPoolRing.removeRef(memPool);
      memPool->modeMemoryPool = NULL;
    }
  }

  void modeMemoryPool_t::dontUseRefs() {
    modeMemoryRing.dontUseRefs();
    memoryPoolRing.dontUseRefs();
  }

  void modeMemoryPool_t::addMemoryPoolRef(memoryPool *memPool) {
    memoryPoolRing.addRef(memPool);
  }

  void modeMemoryPool_t::removeMemoryPoolRef(memoryPool *memPool) {
    memoryPoolRing.removeRef(memPool);
  }

  void modeMemoryPool_t::addModeMemoryRef(modeMemory_t *mem) {
    modeMemoryRing.addRef(mem);

    /*Find how much of this mem is a new reservation*/
    dim_t lo = mem->offset;
    dim_t hi = lo + mem->size;
    for (modeMemory_t* m : reservations) {
      const dim_t mlo = m->offset;
      const dim_t mhi = m->offset+m->size;
      if (mlo>=hi) break;
      if (mhi<=lo) continue;

      if (mlo<=lo && mhi>=hi) {
        hi=lo;
      } else {
        hi = std::min(hi, mhi);
        lo = std::max(lo, mlo);
      }
      if (lo==hi) break;
    }

    /*Add this mem to the reservation list*/
    reservations.emplace(mem);
    reserved += hi-lo;
  }

  void modeMemoryPool_t::removeModeMemoryRef(modeMemory_t *mem) {
    modeMemoryRing.removeRef(mem);

    /*Remove this mem from the reservation list*/
    auto pos = reservations.find(mem);
    reservations.erase(pos);

    /*Find how much of this mem is removed from reserved space*/
    dim_t lo = mem->offset;
    dim_t hi = lo + mem->size;
    for (modeMemory_t* m : reservations) {
      const dim_t mlo = m->offset;
      const dim_t mhi = m->offset+m->size;
      if (mlo>=hi) break;
      if (mhi<=lo) continue;

      if (mlo<=lo && mhi>=hi) {
        hi=lo;
      } else {
        hi = std::min(hi, mhi);
        lo = std::max(lo, mlo);
      }
      if (lo==hi) break;
    }
    reserved -= hi-lo;

    /*prevent mem from deleting this buffer*/
    mem->modeBuffer=NULL;
  }

  bool modeMemoryPool_t::needsFree() const {
    return memoryPoolRing.needsFree();
  }

  modeMemory_t* modeMemoryPool_t::reserve(const udim_t bytes) {
    /*If pool is too small, resize and put the new reservation at the end*/
    if (reserved+bytes>size) {
      resize(reserved+bytes);
      return slice(reserved, bytes);
    }

    /*If pool is empty, put reservation at the beginning*/
    if (reservations.size()==0) {
      return slice(0, bytes);
    }

    /*Look for a unreserved region which fits request*/
    dim_t offset=0;
    dim_t hi = bytes;
    for (modeMemory_t* m : reservations) {
      const dim_t mlo = m->offset;
      const dim_t mhi = m->offset+m->size;
      if (mlo>=offset+bytes) break; /*Found an suitable empty space*/

      offset = std::max(offset, mhi); /*Shift the potential region*/
    }

    if (offset+bytes<=size) {
      return slice(offset, bytes);
    } else {
      resize(reserved+bytes);
      return slice(reserved, bytes);
    }
  }
}
