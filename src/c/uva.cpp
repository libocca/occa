#include <occa/internal/c/types.hpp>
#include <occa/c/uva.h>
#include <occa/utils/uva.hpp>

OCCA_START_EXTERN_C

bool occaIsManaged(void *ptr) {
  return occa::isManaged(ptr);
}

void occaStartManaging(void *ptr) {
  occa::startManaging(ptr);
}

void occaStopManaging(void *ptr) {
  occa::stopManaging(ptr);
}

void occaSyncToDevice(void *ptr,
                      const occaUDim_t bytes) {
  occa::syncToDevice(ptr, bytes);
}

void occaSyncToHost(void *ptr,
                    const occaUDim_t bytes) {
  occa::syncToHost(ptr, bytes);
}

bool occaNeedsSync(void *ptr) {
  return occa::needsSync(ptr);
}

void occaSync(void *ptr) {
  occa::sync(ptr);
}

void occaDontSync(void *ptr) {
  occa::dontSync(ptr);
}

void occaFreeUvaPtr(void *ptr) {
  occa::freeUvaPtr(ptr);
}

OCCA_END_EXTERN_C
