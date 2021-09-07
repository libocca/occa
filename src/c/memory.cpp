#include <occa/internal/c/types.hpp>
#include <occa/c/memory.h>

OCCA_START_EXTERN_C

bool occaMemoryIsInitialized(occaMemory memory) {
  return occa::c::memory(memory).isInitialized();
}

void* occaMemoryPtr(occaMemory memory) {
  occa::memory mem = occa::c::memory(memory);
  return mem.ptr();
}

occaDevice occaMemoryGetDevice(occaMemory memory) {
  return occa::c::newOccaType(
    occa::c::memory(memory).getDevice()
  );
}

occaJson occaMemoryGetProperties(occaMemory memory) {
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

void occaCopyMemToMem(occaMemory dest, occaMemory src,
                      const occaUDim_t bytes,
                      const occaUDim_t destOffset,
                      const occaUDim_t srcOffset,
                      occaJson props) {

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
                 occa::c::json(props));
  }
}

void occaCopyPtrToMem(occaMemory dest, const void *src,
                      const occaUDim_t bytes,
                      const occaUDim_t offset,
                      occaJson props) {

  occa::memory dest_ = occa::c::memory(dest);

  if (occa::c::isDefault(props)) {
    occa::memcpy(dest_, src,
                 bytes,
                 offset);
  } else {
    occa::memcpy(dest_, src,
                 bytes,
                 offset,
                 occa::c::json(props));
  }
}

void occaCopyMemToPtr(void *dest, occaMemory src,
                      const occaUDim_t bytes,
                      const occaUDim_t offset,
                      occaJson props) {

  occa::memory src_ = occa::c::memory(src);

  if (occa::c::isDefault(props)) {
    occa::memcpy(dest, src_,
                 bytes,
                 offset);
  } else {
    occa::memcpy(dest, src_,
                 bytes,
                 offset,
                 occa::c::json(props));
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
