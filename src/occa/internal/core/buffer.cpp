#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/modes/serial/buffer.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  modeBuffer_t::modeBuffer_t(modeDevice_t *modeDevice_,
                             udim_t size_,
                             const occa::json &properties_) :
    properties(properties_),
    ptr(NULL),
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

      delete mem;
    }

    // Remove ref from device
    if (modeDevice) {
      if (!isWrapped) {
        modeDevice->bytesAllocated -= size;
      }

      modeDevice->removeMemoryRef(this);
    }
    size = 0;
    isWrapped = false;
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
}
