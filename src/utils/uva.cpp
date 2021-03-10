#include <map>

#include <occa/core/base.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/utils/misc.hpp>
#include <occa/internal/utils/uva.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/utils/uva.hpp>

namespace occa {
  occa::modeMemory_t* uvaToMemory(void *ptr) {
    if (!ptr) {
      return NULL;
    }
    ptrRangeMap::iterator it = uvaMap.find(ptr);
    return (it == uvaMap.end()) ? NULL : it->second;
  }

  bool isManaged(void *ptr) {
    occa::modeMemory_t *mem = uvaToMemory(ptr);
    if (mem) {
      return (mem->memInfo & uvaFlag::isManaged);
    }
    return false;
  }

  void startManaging(void *ptr) {
    occa::modeMemory_t *mem = uvaToMemory(ptr);
    if (mem) {
      mem->memInfo |= uvaFlag::isManaged;
    }
  }

  void stopManaging(void *ptr) {
    occa::modeMemory_t *mem = uvaToMemory(ptr);
    if (mem) {
      mem->memInfo &= ~uvaFlag::isManaged;
    }
  }

  void syncToDevice(void *ptr, const udim_t bytes) {
    occa::modeMemory_t *mem = uvaToMemory(ptr);
    if (mem) {
      syncMemToDevice(mem, bytes, ptrDiff(mem->uvaPtr, ptr));
    }
  }

  void syncToHost(void *ptr, const udim_t bytes) {
    occa::modeMemory_t *mem = uvaToMemory(ptr);
    if (mem) {
      syncMemToHost(mem, bytes, ptrDiff(mem->uvaPtr, ptr));
    }
  }

  void syncMemToDevice(occa::modeMemory_t *mem,
                       const udim_t bytes,
                       const udim_t offset) {

    if (mem) {
      occa::memory(mem).syncToDevice(bytes, offset);
    }
  }

  void syncMemToHost(occa::modeMemory_t *mem,
                     const udim_t bytes,
                     const udim_t offset) {

    if (mem) {
      occa::memory(mem).syncToHost(bytes, offset);
    }
  }

  bool needsSync(void *ptr) {
    occa::modeMemory_t *mem = uvaToMemory(ptr);
    return mem ? mem->isStale() : false;
  }

  void sync(void *ptr) {
    occa::modeMemory_t *mem = uvaToMemory(ptr);
    if (mem) {
      if (mem->inDevice()) {
        syncMemToHost(mem);
      } else {
        syncMemToDevice(mem);
      }
    }
  }

  void dontSync(void *ptr) {
    removeFromStaleMap(ptr);
  }

  void freeUvaPtr(void *ptr) {
    if (!ptr) {
      return;
    }
    modeMemory_t *modeMemory = uvaToMemory(ptr);
    if (modeMemory) {
      occa::memory(modeMemory).free();
      return;
    }
    OCCA_FORCE_ERROR("Freeing a non-uva pointer");
  }
}
