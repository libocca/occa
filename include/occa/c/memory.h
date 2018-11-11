#ifndef OCCA_C_MEMORY_HEADER
#define OCCA_C_MEMORY_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

OCCA_LFUNC int OCCA_RFUNC occaMemoryIsInitialized(occaMemory memory);

OCCA_LFUNC void* OCCA_RFUNC occaMemoryPtr(occaMemory memory,
                                          occaProperties props);

OCCA_LFUNC occaDevice OCCA_RFUNC occaMemoryGetDevice(occaMemory memory);

OCCA_LFUNC occaProperties OCCA_RFUNC occaMemoryGetProperties(occaMemory memory);

OCCA_LFUNC occaUDim_t OCCA_RFUNC occaMemorySize(occaMemory memory);

OCCA_LFUNC occaMemory OCCA_RFUNC occaMemorySlice(occaMemory memory,
                                                 const occaDim_t offset,
                                                 const occaDim_t bytes);

//---[ UVA ]----------------------------
OCCA_LFUNC int OCCA_RFUNC occaMemoryIsManaged(occaMemory memory);

OCCA_LFUNC int OCCA_RFUNC occaMemoryInDevice(occaMemory memory);

OCCA_LFUNC int OCCA_RFUNC occaMemoryIsStale(occaMemory memory);

OCCA_LFUNC void OCCA_RFUNC occaMemoryStartManaging(occaMemory memory);

OCCA_LFUNC void OCCA_RFUNC occaMemoryStopManaging(occaMemory memory);

OCCA_LFUNC void OCCA_RFUNC occaMemorySyncToDevice(occaMemory memory,
                                                  const occaDim_t bytes,
                                                  const occaDim_t offset);

OCCA_LFUNC void OCCA_RFUNC occaMemorySyncToHost(occaMemory memory,
                                                const occaDim_t bytes,
                                                const occaDim_t offset);
//======================================

OCCA_LFUNC void OCCA_RFUNC occaMemcpy(void *dest,
                                      const void *src,
                                      const occaUDim_t bytes,
                                      occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaCopyMemToMem(occaMemory dest, occaMemory src,
                                            const occaUDim_t bytes,
                                            const occaUDim_t destOffset,
                                            const occaUDim_t srcOffset,
                                            occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaCopyPtrToMem(occaMemory dest,
                                            const void *src,
                                            const occaUDim_t bytes,
                                            const occaUDim_t offset,
                                            occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaCopyMemToPtr(void *dest,
                                            occaMemory src,
                                            const occaUDim_t bytes,
                                            const occaUDim_t offset,
                                            occaProperties props);

OCCA_LFUNC occaMemory OCCA_RFUNC occaMemoryClone(occaMemory memory);

OCCA_LFUNC void OCCA_RFUNC occaMemoryDetach(occaMemory memory);

OCCA_LFUNC occaMemory OCCA_RFUNC occaWrapCpuMemory(occaDevice device,
                                                   void *ptr,
                                                   occaUDim_t bytes,
                                                   occaProperties props);

OCCA_END_EXTERN_C

#endif
