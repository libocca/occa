#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/buffer.hpp>
#include <occa/internal/modes/serial/memory.hpp>

namespace occa {

  modeMemory_t::modeMemory_t(modeBuffer_t *modeBuffer_,
                             udim_t size_, dim_t offset_) :
    modeBuffer(modeBuffer_),
    ptr(NULL),
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
  }

  void modeMemory_t::addMemoryRef(memory *mem) {
    memoryRing.addRef(mem);
  }

  void modeMemory_t::removeMemoryRef(memory *mem) {
    memoryRing.removeRef(mem);
  }

  void modeMemory_t::removeModeMemoryRef() {
    if (modeBuffer == NULL) return;

    modeBuffer->removeModeMemoryRef(this);

    if (modeBuffer->needsFree()) {
      delete modeBuffer;
    }
    modeBuffer = NULL;
  }

  void modeMemory_t::detach() {
    if (modeBuffer == NULL) return;
    modeBuffer->detach();
  }

  bool modeMemory_t::needsFree() const {
    return memoryRing.needsFree();
  }

  modeDevice_t* modeMemory_t::getModeDevice() const {
    return (modeBuffer
            ? modeBuffer->modeDevice
            : nullptr);
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

    return modeBuffer->slice(offset+offset_, bytes);
  }
}
