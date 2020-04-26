#ifndef OCCA_C_UVA_HEADER
#define OCCA_C_UVA_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

OCCA_LFUNC bool OCCA_RFUNC occaIsManaged(void *ptr);
OCCA_LFUNC void OCCA_RFUNC occaStartManaging(void *ptr);
OCCA_LFUNC void OCCA_RFUNC occaStopManaging(void *ptr);

OCCA_LFUNC void OCCA_RFUNC occaSyncToDevice(void *ptr,
                                            const occaUDim_t bytes);
OCCA_LFUNC void OCCA_RFUNC occaSyncToHost(void *ptr,
                                          const occaUDim_t bytes);

OCCA_LFUNC bool OCCA_RFUNC occaNeedsSync(void *ptr);
OCCA_LFUNC void OCCA_RFUNC occaSync(void *ptr);
OCCA_LFUNC void OCCA_RFUNC occaDontSync(void *ptr);

OCCA_LFUNC void OCCA_RFUNC occaFreeUvaPtr(void *ptr);

OCCA_END_EXTERN_C

#endif
