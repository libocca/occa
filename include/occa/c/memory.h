#ifndef OCCA_C_MEMORY_HEADER
#define OCCA_C_MEMORY_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

bool occaMemoryIsInitialized(occaMemory memory);

void* occaMemoryPtr(occaMemory memory);

occaDevice occaMemoryGetDevice(occaMemory memory);

occaJson occaMemoryGetProperties(occaMemory memory);

occaUDim_t occaMemorySize(occaMemory memory);

occaMemory occaMemorySlice(occaMemory memory,
                           const occaDim_t offset,
                           const occaDim_t bytes);

//---[ UVA ]----------------------------
bool occaMemoryIsManaged(occaMemory memory);

bool occaMemoryInDevice(occaMemory memory);

bool occaMemoryIsStale(occaMemory memory);

void occaMemoryStartManaging(occaMemory memory);

void occaMemoryStopManaging(occaMemory memory);

void occaMemorySyncToDevice(occaMemory memory,
                            const occaDim_t bytes,
                            const occaDim_t offset);

void occaMemorySyncToHost(occaMemory memory,
                          const occaDim_t bytes,
                          const occaDim_t offset);
//======================================

void occaMemcpy(void *dest,
                const void *src,
                const occaUDim_t bytes,
                occaJson props);

void occaCopyMemToMem(occaMemory dest, occaMemory src,
                      const occaUDim_t bytes,
                      const occaUDim_t destOffset,
                      const occaUDim_t srcOffset,
                      occaJson props);

void occaCopyPtrToMem(occaMemory dest,
                      const void *src,
                      const occaUDim_t bytes,
                      const occaUDim_t offset,
                      occaJson props);

void occaCopyMemToPtr(void *dest,
                      occaMemory src,
                      const occaUDim_t bytes,
                      const occaUDim_t offset,
                      occaJson props);

occaMemory occaMemoryClone(occaMemory memory);

void occaMemoryDetach(occaMemory memory);

occaMemory occaWrapCpuMemory(occaDevice device,
                             void *ptr,
                             occaUDim_t bytes,
                             occaJson props);

OCCA_END_EXTERN_C

#endif
