#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/modes/serial/memoryPool.hpp>

namespace occa {

  using experimental::memoryPool;

  modeMemoryPool_t::modeMemoryPool_t(modeDevice_t *modeDevice_,
                                     const occa::json &properties_) :
    modeBuffer_t(modeDevice_, 0, properties_),
    reserved(0),
    buffer(nullptr) {
  }

  modeMemoryPool_t::~modeMemoryPool_t() {
    // NULL all wrappers
    while (memoryPoolRing.head) {
      memoryPool *memPool = (memoryPool*) memoryPoolRing.head;
      memoryPoolRing.removeRef(memPool);
      memPool->modeMemoryPool = NULL;
    }
    if (buffer) delete buffer;
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
      if (mlo >= hi) break;
      if (mhi <= lo) continue;

      if (mlo <= lo && mhi >= hi) {
        hi = lo;
      } else {
        hi = std::min(hi, mhi);
        lo = std::max(lo, mlo);
      }
      if (lo == hi) break;
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
      if (mlo >= hi) break;
      if (mhi <= lo) continue;

      if (mlo <= lo && mhi >= hi) {
        hi = lo;
      } else {
        hi = std::min(hi, mhi);
        lo = std::max(lo, mlo);
      }
      if (lo == hi) break;
    }
    reserved -= hi-lo;

    /*prevent mem from deleting this buffer*/
    mem->modeBuffer = nullptr;
  }

  bool modeMemoryPool_t::needsFree() const {
    return memoryPoolRing.needsFree();
  }

  modeMemory_t* modeMemoryPool_t::reserve(const udim_t bytes) {
    /*If pool is too small, resize and put the new reservation at the end*/
    if (reserved + bytes > size) {
      resize(reserved + bytes);
      return slice(reserved, bytes);
    }

    /*If pool is empty, put reservation at the beginning*/
    if (reservations.size()==0) {
      return slice(0, bytes);
    }

    /*Look for a unreserved region which fits request*/
    dim_t offset = 0;
    for (modeMemory_t* m : reservations) {
      const dim_t mlo = m->offset;
      const dim_t mhi = m->offset+m->size;
      if (mlo >= static_cast<dim_t>(offset + bytes)) break; /*Found an suitable empty space*/

      offset = std::max(offset, mhi); /*Shift the potential region*/
    }

    if (offset + bytes <= size) {
      return slice(offset, bytes);
    } else {
      resize(reserved + bytes);
      return slice(reserved, bytes);
    }
  }

  void modeMemoryPool_t::resize(const udim_t bytes) {

    OCCA_ERROR("Cannot resize memoryPool below current usage"
               "(reserved: " << reserved << ", bytes: " << bytes << ")",
               reserved <= bytes);

    if (size == bytes) return; /*Nothing to do*/

    if (reservations.size() == 0) {
      /*
      If there are no outstanding reservations,
      destroy the allocation and re-make it
      */
      if (buffer) delete buffer;
      modeDevice->bytesAllocated -= size;

      buffer = makeBuffer();
      buffer->malloc(bytes);
      size = bytes;

      modeDevice->bytesAllocated += bytes;
      modeDevice->maxBytesAllocated = std::max(
        modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
      );

    } else {
      /*
      There are currently reservations.
      Make a new allocation and migrate reserved space to new allocation
      packing the space in the process
      */
      modeBuffer_t* newBuffer = makeBuffer();
      newBuffer->malloc(bytes);

      modeDevice->bytesAllocated += bytes;
      modeDevice->maxBytesAllocated = std::max(
        modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
      );

      /*Loop through the reservation list*/
      auto it = reservations.begin();
      modeMemory_t* m = *it;
      dim_t lo = m->offset;    /*Start point of current block*/
      dim_t hi = lo + m->size; /*End point of current block*/
      dim_t offset = 0;
      setPtr(m, newBuffer, offset);
      do {

        it++;

        if (it == reservations.end()) {
          /*If this reservation is the last one, copy the block and we're done*/
          memcpy(newBuffer, offset, buffer, lo, hi - lo);
        } else {
          /*Look at next reservation*/
          m = *it;
          const dim_t mlo = m->offset;
          const dim_t mhi = m->offset + m->size;
          if (mlo > hi) {
            /*
            If the start point of the next reservation is in a new block
            copy the last block to the new allocation
            */
            memcpy(newBuffer, offset, buffer, lo, hi - lo);

            /*Increment offset, and track start/end of current block*/
            offset += hi - lo;
            lo = mlo;
            hi = mhi;
          } else {
            /*
            Reservation is in the same block.
            Extend end point of current block
            */
            hi = std::max(hi, mhi);
          }
          /*Update the buffer of this reservation*/
          setPtr(m, newBuffer, m->offset - (lo - offset));
        }
      } while (it != reservations.end());

      /*Clean up old buffer*/
      delete buffer;
      modeDevice->bytesAllocated -= size;

      buffer = newBuffer;
      size = bytes;
    }
  }
}
