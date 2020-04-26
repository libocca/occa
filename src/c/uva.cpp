#include <occa/c/types.hpp>
#include <occa/c/uva.h>
#include <occa/tools/uva.hpp>

OCCA_START_EXTERN_C

OCCA_LFUNC bool OCCA_RFUNC occaIsManaged(void *ptr) {
  return occa::isManaged(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaStartManaging(void *ptr) {
  occa::startManaging(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaStopManaging(void *ptr) {
  occa::stopManaging(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaSyncToDevice(void *ptr,
                                            const occaUDim_t bytes) {
  occa::syncToDevice(ptr, bytes);
}

OCCA_LFUNC void OCCA_RFUNC occaSyncToHost(void *ptr,
                                          const occaUDim_t bytes) {
  occa::syncToHost(ptr, bytes);
}

OCCA_LFUNC bool OCCA_RFUNC occaNeedsSync(void *ptr) {
  return occa::needsSync(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaSync(void *ptr) {
  occa::sync(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaDontSync(void *ptr) {
  occa::dontSync(ptr);
}

void OCCA_RFUNC occaFreeUvaPtr(void *ptr) {
  occa::freeUvaPtr(ptr);
}

OCCA_END_EXTERN_C
