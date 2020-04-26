#include <occa/c/types.hpp>
#include <occa/c/memory.h>

OCCA_START_EXTERN_C

bool OCCA_RFUNC occaMemoryIsInitialized(occaMemory memory) {
  return occa::c::memory(memory).isInitialized();
}

void* OCCA_RFUNC occaMemoryPtr(occaMemory memory,
                               occaProperties props) {
  occa::memory mem = occa::c::memory(memory);
  if (occa::c::isDefault(props)) {
    return mem.ptr();
  }
  return mem.ptr(occa::c::properties(props));
}

occaDevice OCCA_RFUNC occaMemoryGetDevice(occaMemory memory) {
  return occa::c::newOccaType(
    occa::c::memory(memory).getDevice()
  );
}

occaProperties OCCA_RFUNC occaMemoryGetProperties(occaMemory memory) {
  return occa::c::newOccaType(
    occa::c::memory(memory).properties(),
    false
  );
}

occaUDim_t OCCA_RFUNC occaMemorySize(occaMemory memory) {
  return occa::c::memory(memory).size();
}

occaMemory OCCA_RFUNC occaMemorySlice(occaMemory memory,
                                      const occaDim_t offset,
                                      const occaDim_t bytes) {
  occa::memory memSlice = occa::c::memory(memory).slice(offset, bytes);
  memSlice.dontUseRefs();
  return occa::c::newOccaType(memSlice);
}

//---[ UVA ]----------------------------
bool OCCA_RFUNC occaMemoryIsManaged(occaMemory memory) {
  return (int) occa::c::memory(memory).isManaged();
}

bool OCCA_RFUNC occaMemoryInDevice(occaMemory memory) {
  return (int) occa::c::memory(memory).inDevice();
}

bool OCCA_RFUNC occaMemoryIsStale(occaMemory memory) {
  return (int) occa::c::memory(memory).isStale();
}

void OCCA_RFUNC occaMemoryStartManaging(occaMemory memory) {
  occa::c::memory(memory).startManaging();
}

void OCCA_RFUNC occaMemoryStopManaging(occaMemory memory) {
  occa::c::memory(memory).stopManaging();
}

void OCCA_RFUNC occaMemorySyncToDevice(occaMemory memory,
                                       const occaDim_t bytes,
                                       const occaDim_t offset) {

  occa::c::memory(memory).syncToDevice(bytes, offset);
}

void OCCA_RFUNC occaMemorySyncToHost(occaMemory memory,
                                     const occaDim_t bytes,
                                     const occaDim_t offset) {

  occa::c::memory(memory).syncToHost(bytes, offset);
}
//======================================

void OCCA_RFUNC occaMemcpy(void *dest, const void *src,
                           const occaUDim_t bytes,
                           occaProperties props) {
  if (occa::c::isDefault(props)) {
    occa::memcpy(dest, src, bytes);
  } else {
    occa::memcpy(dest, src, bytes,
                 occa::c::properties(props));
  }
}

void OCCA_RFUNC occaCopyMemToMem(occaMemory dest, occaMemory src,
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

void OCCA_RFUNC occaCopyPtrToMem(occaMemory dest, const void *src,
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

void OCCA_RFUNC occaCopyMemToPtr(void *dest, occaMemory src,
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

occaMemory OCCA_RFUNC occaMemoryClone(occaMemory memory) {
  occa::memory clone = occa::c::memory(memory).clone();
  clone.dontUseRefs();
  return occa::c::newOccaType(clone);
}

void OCCA_RFUNC occaMemoryDetach(occaMemory memory) {
  occa::c::memory(memory).detach();
}

occaMemory OCCA_RFUNC occaWrapCpuMemory(occaDevice device,
                                        void *ptr,
                                        occaUDim_t bytes,
                                        occaProperties props) {
  occa::device device_ = (
    occa::c::isDefault(device)
    ? occa::getDevice()
    : occa::c::device(device)
  );

  occa::memory mem;
  if (occa::c::isDefault(props)) {
    mem = occa::cpu::wrapMemory(device_,
                                ptr,
                                bytes);
  } else {
    mem = occa::cpu::wrapMemory(device_,
                                ptr,
                                bytes,
                                occa::c::properties(props));
  }
  mem.dontUseRefs();

  return occa::c::newOccaType(mem);
}

OCCA_END_EXTERN_C
