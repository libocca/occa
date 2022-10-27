#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/modes/serial/memoryPool.hpp>

namespace occa {

  using experimental::memoryPool;

  modeMemoryPool_t::modeMemoryPool_t(modeDevice_t *modeDevice_,
                                     const occa::json &properties_) :
    modeBuffer_t(modeDevice_, 0, properties_),
    alignment(128),
    reserved(0),
    buffer(nullptr) {
    verbose = properties_.get("verbose", false);
  }

  modeMemoryPool_t::~modeMemoryPool_t() {
    // NULL all wrappers
    while (memoryPoolRing.head) {
      memoryPool *memPool = (memoryPool*) memoryPoolRing.head;
      memoryPoolRing.removeRef(memPool);
      memPool->modeMemoryPool = NULL;
    }
    if (buffer) delete buffer;
    size=0;
  }

  void modeMemoryPool_t::dontUseRefs() {
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
    dim_t lo = (mem->offset / alignment) * alignment; //Round down to alignment
    dim_t hi = ((mem->offset + mem->size + alignment - 1)
                / alignment) * alignment; //Round up
    for (modeMemory_t* m : reservations) {
      const dim_t mlo = (m->offset / alignment) * alignment;
      const dim_t mhi = ((m->offset + m->size + alignment - 1)
                        / alignment) * alignment;
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
    dim_t lo = (mem->offset / alignment) * alignment; //Round down to alignment
    dim_t hi = ((mem->offset + mem->size + alignment - 1)
                / alignment) * alignment; //Round up
    for (modeMemory_t* m : reservations) {
      const dim_t mlo = (m->offset / alignment) * alignment;
      const dim_t mhi = ((m->offset + m->size + alignment - 1)
                        / alignment) * alignment;
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
  }

  bool modeMemoryPool_t::needsFree() const {
    return memoryPoolRing.needsFree();
  }

  udim_t modeMemoryPool_t::numReservations() const {
    return reservations.size();
  }

  modeMemory_t* modeMemoryPool_t::reserve(const udim_t bytes) {

    const udim_t alignedBytes = ((bytes + alignment - 1) / alignment) * alignment;

    /*If pool is too small, resize and put the new reservation at the end*/
    if (reserved + bytes > size) {
      resize(reserved + alignedBytes);
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
      const dim_t mhi = ((m->offset + m->size + alignment - 1)
                        / alignment) * alignment; //Round up upper limit
      if (mlo >= static_cast<dim_t>(offset + bytes)) break; /*Found a suitable empty space*/

      offset = std::max(offset, mhi); /*Shift the potential region*/
    }

    if (offset + bytes <= size) {
      return slice(offset, bytes);
    } else {
      resize(reserved + alignedBytes);
      return slice(reserved, bytes);
    }
  }

  void modeMemoryPool_t::resize(const udim_t bytes) {

    OCCA_ERROR("Cannot resize memoryPool below current usage"
               "(reserved: " << reserved << ", bytes: " << bytes << ")",
               reserved <= bytes);

    if (size == bytes) return; /*Nothing to do*/

    const udim_t alignedBytes = ((bytes + alignment - 1) / alignment) * alignment;

    if (verbose) {
      io::stdout << "MemoryPool: Resizing to " << alignedBytes << " bytes\n";
    }

    if (reservations.size() == 0) {
      /*
      If there are no outstanding reservations,
      destroy the allocation and re-make it
      */
      if (buffer) delete buffer;

      buffer = makeBuffer();
      buffer->malloc(alignedBytes);
      size = alignedBytes;

      modeDevice->bytesAllocated += alignedBytes;
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
      newBuffer->malloc(alignedBytes);

      modeDevice->bytesAllocated += alignedBytes;
      modeDevice->maxBytesAllocated = std::max(
        modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
      );

      /*Loop through the reservation list*/
      auto it = reservations.begin();
      modeMemory_t* m = *it;
      dim_t lo = m->offset;    /*Start point of current block*/
      dim_t hi = lo + m->size; /*End point of current block*/
      dim_t offset = 0;
      udim_t newReserved = 0;
      setPtr(m, newBuffer, offset);
      do {

        it++;

        if (it == reservations.end()) {
          /*If this reservation is the last one, copy the block and we're done*/
          memcpy(newBuffer, offset, buffer, lo, hi - lo);
          newReserved += ((hi - lo + alignment - 1) / alignment) * alignment;
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
            const udim_t reservationSize = ((hi - lo + alignment - 1) / alignment) * alignment;
            newReserved += reservationSize;

            /*Increment offset, and track start/end of current block*/
            offset += reservationSize;
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

      buffer = newBuffer;
      size = alignedBytes;
      reserved = newReserved;
    }
  }

  void modeMemoryPool_t::setAlignment(const udim_t newAlignment) {

    OCCA_ERROR("Cannot set memoryPool alignment to zero bytes",
               newAlignment != 0);

    if (alignment == newAlignment) return; /*Nothing to do*/

    if (reservations.size() != 0) {
      /*
      There are currently reservations.
      Figure out the size of the new buffer needed
      */
      /*Loop through the reservation list*/
      auto it = reservations.begin();
      modeMemory_t* m = *it;
      dim_t lo = m->offset;    /*Start point of current block*/
      dim_t hi = lo + m->size; /*End point of current block*/
      udim_t newReserved = 0;
      do {
        it++;
        if (it == reservations.end()) {
          newReserved += ((hi - lo + newAlignment - 1) / newAlignment) * newAlignment;
        } else {
          /*Look at next reservation*/
          m = *it;
          const dim_t mlo = m->offset;
          const dim_t mhi = m->offset + m->size;
          if (mlo > hi) {
            /*
            If the start point of the next reservation is in a new block
            */
            const udim_t reservationSize = ((hi - lo + newAlignment - 1) / newAlignment) * newAlignment;
            newReserved += reservationSize;

            /*Track start/end of current block*/
            lo = mlo;
            hi = mhi;
          } else {
            /*
            Reservation is in the same block.
            Extend end point of current block
            */
            hi = std::max(hi, mhi);
          }
        }
      } while (it != reservations.end());

      /*Make a new buffer*/
      modeBuffer_t* newBuffer = makeBuffer();
      newBuffer->malloc(newReserved);

      modeDevice->bytesAllocated += newReserved;
      modeDevice->maxBytesAllocated = std::max(
        modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
      );

      /*Loop through the reservation list and migrate to new alignment*/
      it = reservations.begin();
      m = *it;
      lo = m->offset;    /*Start point of current block*/
      hi = lo + m->size; /*End point of current block*/
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
            const udim_t reservationSize = ((hi - lo + newAlignment - 1) / newAlignment) * newAlignment;

            /*Increment offset, and track start/end of current block*/
            offset += reservationSize;
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

      buffer = newBuffer;
      size = newReserved;
      reserved = newReserved;
    }

    alignment = newAlignment;
  }
}
