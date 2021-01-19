#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/utils/uva.hpp>

namespace occa {
  modeMemory_t::modeMemory_t(modeDevice_t *modeDevice_,
                             udim_t size_,
                             const occa::json &properties_) :
    memInfo(uvaFlag::none),
    properties(properties_),
    ptr(NULL),
    uvaPtr(NULL),
    modeDevice(modeDevice_),
    dtype_(&dtype::byte),
    size(size_),
    isOrigin(true) {
    modeDevice->addMemoryRef(this);
  }

  modeMemory_t::~modeMemory_t() {
    // NULL all wrappers
    while (memoryRing.head) {
      memory *mem = (memory*) memoryRing.head;
      memoryRing.removeRef(mem);
      mem->modeMemory = NULL;
    }
    // Remove ref from device
    if (modeDevice) {
      modeDevice->removeMemoryRef(this);
    }
  }

  void* modeMemory_t::getPtr() {
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
}
