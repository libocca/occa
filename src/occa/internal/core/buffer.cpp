#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/modes/serial/buffer.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/utils/uva.hpp>

namespace occa {
  modeBuffer_t::modeBuffer_t(modeDevice_t *modeDevice_,
                             udim_t size_,
                             const occa::json &properties_) :
    properties(properties_),
    ptr(NULL),
    uvaPtr(NULL),
    modeDevice(modeDevice_),
    size(size_),
    isWrapped(false) {
    modeDevice->addMemoryRef(this);
  }

  modeBuffer_t::~modeBuffer_t() {
    // destroy all slices
    while (modeMemoryRing.head) {
      modeMemory_t *mem = (modeMemory_t*) modeMemoryRing.head;
      removeModeMemoryRef(mem);
      mem->modeBuffer = NULL;

      if (uvaPtr) {
        void *uvaPtr_ = mem->uvaPtr;

        uvaMap.erase(uvaPtr_);
        modeDevice->uvaMap.erase(uvaPtr_);
      }

      delete mem;
    }

    // CPU case where memory is shared
    if (uvaPtr && (modeDevice->hasSeparateMemorySpace())) {
      sys::free(uvaPtr);
    }
    uvaPtr = NULL;

    // Remove ref from device
    if (modeDevice) {
      if (!isWrapped)
        modeDevice->bytesAllocated -= size;

      modeDevice->removeMemoryRef(this);
    }
    size = 0;
    isWrapped = false;
  }

  void modeBuffer_t::dontUseRefs() {
    modeMemoryRing.dontUseRefs();
  }

  void modeBuffer_t::addModeMemoryRef(modeMemory_t *mem) {
    modeMemoryRing.addRef(mem);
  }

  void modeBuffer_t::removeModeMemoryRef(modeMemory_t *mem) {
    modeMemoryRing.removeRef(mem);
  }

  bool modeBuffer_t::needsFree() const {
    return modeMemoryRing.needsFree();
  }

  void modeBuffer_t::setupUva() {
    if ( !(modeDevice->hasSeparateMemorySpace()) ) {
      uvaPtr = ptr;
    } else {
      uvaPtr = (char*) sys::malloc(size);
    }

    //set uvaPtr in all slices
    if (!modeMemoryRing.head) return;

    modeMemory_t *head = (modeMemory_t*) modeMemoryRing.head;
    modeMemory_t *mem  = head;
    do {
      mem->uvaPtr = uvaPtr + mem->offset;

      ptrRange range;
      range.start = mem->uvaPtr;
      range.end   = (range.start + mem->size);

      uvaMap[range] = mem;
      mem->getModeDevice()->uvaMap[range] = mem;

      // Needed for kernelArg.void_ -> modeMemory checks
      if (mem->uvaPtr != mem->ptr) {
        uvaMap[mem->ptr] = mem;
      }

      mem = (modeMemory_t*) mem->rightRingEntry;

    } while (mem != head);
  }
}
