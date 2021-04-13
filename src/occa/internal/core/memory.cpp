#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/buffer.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/utils/uva.hpp>

namespace occa {
  modeMemory_t::modeMemory_t(modeBuffer_t *modeBuffer_,
                             udim_t size_, dim_t offset_) :
    memInfo(uvaFlag::none),
    modeBuffer(modeBuffer_),
    ptr(NULL),
    uvaPtr(NULL),
    dtype_(&dtype::byte),
    size(size_),
    offset(offset_) {
    modeBuffer->addModeMemoryRef(this);
  }

  modeMemory_t::~modeMemory_t() {
    // NULL all wrappers
    while (memoryRing.head) {
      memory *mem = (memory*) memoryRing.head;
      memoryRing.removeRef(mem);
      mem->modeMemory = NULL;
    }

    // Remove ref from buffer
    removeModeMemoryRef();
  }

  void* modeMemory_t::getPtr() const {
    return ptr;
  }

  void modeMemory_t::dontUseRefs() {
    memoryRing.dontUseRefs();
    if (modeBuffer) modeBuffer->dontUseRefs();
  }

  void modeMemory_t::addMemoryRef(memory *mem) {
    memoryRing.addRef(mem);
  }

  void modeMemory_t::removeMemoryRef(memory *mem) {
    memoryRing.removeRef(mem);
  }

  void modeMemory_t::removeModeMemoryRef() {
    if (!modeBuffer) {
      return;
    }
    modeBuffer->removeModeMemoryRef(this);
    if (modeBuffer->modeBuffer_t::needsFree()) {
      free();
    }
  }

  void modeMemory_t::detach() {
    if (modeBuffer == NULL) return;

    modeBuffer->detach();

    //deleting the modeBuffer deletes all
    // the modeMemory_t slicing it, and NULLs
    // their wrappers
    delete modeBuffer;
  }

  void modeMemory_t::free() {
    if (modeBuffer == NULL) return;
    delete modeBuffer;
  }

  bool modeMemory_t::needsFree() const {
    return memoryRing.needsFree();
  }

  bool modeMemory_t::isManaged() const {
    return (memInfo & uvaFlag::isManaged);
  }

  bool modeMemory_t::inDevice() const {
    return (memInfo & uvaFlag::inDevice);
  }

  bool modeMemory_t::isStale() const {
    return (memInfo & uvaFlag::isStale);
  }

  void modeMemory_t::setupUva() {
    if (!modeBuffer) {
      return;
    }
    modeBuffer->setupUva();
  }

  modeDevice_t* modeMemory_t::getModeDevice() const {
    return modeBuffer->modeDevice;
  }

  const occa::json& modeMemory_t::properties() const {
    static const occa::json noProperties;
    return (modeBuffer
            ? modeBuffer->properties
            : noProperties);
  }

  modeMemory_t* modeMemory_t::slice(const dim_t offset_,
                                    const udim_t bytes) {

    //quick return if we're not really slicing
    if ((offset_ == 0) && (bytes == size)) return this;

    OCCA_ERROR("ModeMemory not initialized or has been freed",
               modeBuffer != NULL);


    OCCA_ERROR("Cannot have a negative offset (" << offset + offset_ << ")",
               offset + offset_ >= 0);

    modeMemory_t* m = modeBuffer->slice(offset+offset_, bytes);

    if (uvaPtr) {
      m->uvaPtr = (uvaPtr + offset_);

      ptrRange range;
      range.start = m->uvaPtr;
      range.end   = (range.start + m->size);

      uvaMap[range] = m;
      m->getModeDevice()->uvaMap[range] = m;

      // Needed for kernelArg.void_ -> modeMemory checks
      if (m->uvaPtr != m->ptr) {
        uvaMap[m->ptr] = m;
      }
    }

    return m;
  }
}
