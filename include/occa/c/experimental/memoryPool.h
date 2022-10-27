#ifndef OCCA_C_MEMORYPOOL_HEADER
#define OCCA_C_MEMORYPOOL_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

bool occaMemoryPoolIsInitialized(occaMemoryPool memoryPool);

occaDevice occaMemoryPoolGetDevice(occaMemoryPool memoryPool);

const char* occaMemoryPoolMode(occaMemoryPool memoryPool);

occaJson occaMemoryPoolGetProperties(occaMemoryPool memoryPool);

occaUDim_t occaMemoryPoolSize(occaMemoryPool memoryPool);

occaUDim_t occaMemoryPoolReserved(occaMemoryPool memoryPool);

occaUDim_t occaMemoryPoolNumReservations(occaMemoryPool memoryPool);

occaUDim_t occaMemoryPoolAlignment(occaMemoryPool memoryPool);

void occaMemoryPoolResize(occaMemoryPool memoryPool,
                          const occaUDim_t bytes);

void occaMemoryPoolShrinkToFit(occaMemoryPool memoryPool);

occaMemory occaMemoryPoolReserve(occaMemoryPool memoryPool,
                                 const occaUDim_t bytes);

occaMemory occaMemoryPoolTypedReserve(occaMemoryPool memoryPool,
                                      const occaUDim_t entries,
                                      const occaDtype dtype);

void occaMemoryPoolSetAlignment(occaMemoryPool memoryPool,
                                const occaUDim_t alignment);

OCCA_END_EXTERN_C

#endif
