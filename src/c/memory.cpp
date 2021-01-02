#include <occa/internal/c/types.hpp>
#include <occa/c/memory.h>

OCCA_START_EXTERN_C

bool occaMemoryIsInitialized(occaMemory memory) {
  return occa::c::memory(memory).isInitialized();
}

void* occaMemoryPtr(occaMemory memory,
                    occaProperties props) {
  occa::memory mem = occa::c::memory(memory);
  if (occa::c::isDefault(props)) {
    return mem.ptr();
  }
  return mem.ptr(occa::c::properties(props));
}

occaDevice occaMemoryGetDevice(occaMemory memory) {
  return occa::c::newOccaType(
    occa::c::memory(memory).getDevice()
  );
}

occaProperties occaMemoryGetProperties(occaMemory memory) {
  return occa::c::newOccaType(
    occa::c::memory(memory).properties(),
    false
  );
}

occaUDim_t occaMemorySize(occaMemory memory) {
  return occa::c::memory(memory).size();
}

occaMemory occaMemorySlice(occaMemory memory,
                           const occaDim_t offset,
                           const occaDim_t bytes) {
  occa::memory memSlice = occa::c::memory(memory).slice(offset, bytes);
  memSlice.dontUseRefs();
  return occa::c::newOccaType(memSlice);
}

//---[ UVA ]----------------------------
bool occaMemoryIsManaged(occaMemory memory) {
  return (int) occa::c::memory(memory).isManaged();
}

bool occaMemoryInDevice(occaMemory memory) {
  return (int) occa::c::memory(memory).inDevice();
}

bool occaMemoryIsStale(occaMemory memory) {
  return (int) occa::c::memory(memory).isStale();
}

void occaMemoryStartManaging(occaMemory memory) {
  occa::c::memory(memory).startManaging();
}

void occaMemoryStopManaging(occaMemory memory) {
  occa::c::memory(memory).stopManaging();
}

void occaMemorySyncToDevice(occaMemory memory,
                            const occaDim_t bytes,
                            const occaDim_t offset) {

  occa::c::memory(memory).syncToDevice(bytes, offset);
}

void occaMemorySyncToHost(occaMemory memory,
                          const occaDim_t bytes,
                          const occaDim_t offset) {

  occa::c::memory(memory).syncToHost(bytes, offset);
}
//======================================

void occaMemcpy(void *dest, const void *src,
                const occaUDim_t bytes,
                occaProperties props) {
  if (occa::c::isDefault(props)) {
    occa::memcpy(dest, src, bytes);
  } else {
    occa::memcpy(dest, src, bytes,
                 occa::c::properties(props));
  }
}

void occaCopyMemToMem(occaMemory dest, occaMemory src,
                      const occaUDim_t bytes,
                      const occaUDim_t destOffset,
                      const occaUDim_t srcOffset,
                      occaProperties props) {

  occa::memory src_ = occa::c::memory(src);
  occa::memory dest_ = occa::c::memory(dest);

  if (occa::c::isDefault(props)) {
    occa::memcpy(dest_, src_,
                 bytes,
                 destOffset, srcOffset);
  } else {
    occa::memcpy(dest_, src_,
                 bytes,
                 destOffset, srcOffset,
                 occa::c::properties(props));
  }
}

void occaCopyPtrToMem(occaMemory dest, const void *src,
                      const occaUDim_t bytes,
                      const occaUDim_t offset,
                      occaProperties props) {

  occa::memory dest_ = occa::c::memory(dest);

  if (occa::c::isDefault(props)) {
    occa::memcpy(dest_, src,
                 bytes,
                 offset);
  } else {
    occa::memcpy(dest_, src,
                 bytes,
                 offset,
                 occa::c::properties(props));
  }
}

void occaCopyMemToPtr(void *dest, occaMemory src,
                      const occaUDim_t bytes,
                      const occaUDim_t offset,
                      occaProperties props) {

  occa::memory src_ = occa::c::memory(src);

  if (occa::c::isDefault(props)) {
    occa::memcpy(dest, src_,
                 bytes,
                 offset);
  } else {
    occa::memcpy(dest, src_,
                 bytes,
                 offset,
                 occa::c::properties(props));
  }
}

occaMemory occaMemoryClone(occaMemory memory) {
  occa::memory clone = occa::c::memory(memory).clone();
  clone.dontUseRefs();
  return occa::c::newOccaType(clone);
}

void occaMemoryDetach(occaMemory memory) {
  occa::c::memory(memory).detach();
}

OCCA_END_EXTERN_C
