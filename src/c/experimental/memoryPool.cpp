#include <occa/internal/c/types.hpp>
#include <occa/c/dtype.h>
#include <occa/c/memory.h>
#include <occa/c/experimental/memoryPool.h>

OCCA_START_EXTERN_C

bool occaMemoryPoolIsInitialized(occaMemoryPool memoryPool) {
  return occa::c::memoryPool(memoryPool).isInitialized();
}

occaDevice occaMemoryPoolGetDevice(occaMemoryPool memoryPool) {
  return occa::c::newOccaType(
    occa::c::memoryPool(memoryPool).getDevice()
  );
}

occaJson occaMemoryPoolGetProperties(occaMemoryPool memoryPool) {
  return occa::c::newOccaType(
    occa::c::memoryPool(memoryPool).properties(),
    false
  );
}

occaUDim_t occaMemoryPoolSize(occaMemoryPool memoryPool) {
  return occa::c::memoryPool(memoryPool).size();
}

occaUDim_t occaMemoryPoolReserved(occaMemoryPool memoryPool) {
  return occa::c::memoryPool(memoryPool).reserved();
}

occaUDim_t occaMemoryPoolNumReservations(occaMemoryPool memoryPool) {
  return occa::c::memoryPool(memoryPool).numReservations();
}

occaUDim_t occaMemoryPoolAlignment(occaMemoryPool memoryPool) {
  return occa::c::memoryPool(memoryPool).alignment();
}

void occaMemoryPoolResize(occaMemoryPool memoryPool,
                          const occaUDim_t bytes) {
  occa::c::memoryPool(memoryPool).resize(bytes);
}

void occaMemoryPoolShrinkToFit(occaMemoryPool memoryPool) {
  occa::c::memoryPool(memoryPool).shrinkToFit();
}

occaMemory occaMemoryPoolReserve(occaDevice device,
                                 const occaUDim_t bytes) {
  return occaMemoryPoolTypedReserve(device,
                                    bytes,
                                    occaDtypeByte);
}

occaMemory occaMemoryPoolTypedReserve(occaMemoryPool memoryPool,
                                      const occaUDim_t entries,
                                      const occaDtype dtype) {
  occa::experimental::memoryPool memoryPool_ = occa::c::memoryPool(memoryPool);
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  occa::memory memory = memoryPool_.reserve(entries, dtype_);
  memory.dontUseRefs();

  return occa::c::newOccaType(memory);
}

void occaMemoryPoolSetAlignment(occaMemoryPool memoryPool,
                                const occaUDim_t alignment) {
  occa::c::memoryPool(memoryPool).setAlignment(alignment);
}

OCCA_END_EXTERN_C
