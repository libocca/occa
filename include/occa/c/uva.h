#ifndef OCCA_C_UVA_HEADER
#define OCCA_C_UVA_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

bool occaIsManaged(void *ptr);
void occaStartManaging(void *ptr);
void occaStopManaging(void *ptr);

void occaSyncToDevice(void *ptr,
                      const occaUDim_t bytes);
void occaSyncToHost(void *ptr,
                    const occaUDim_t bytes);

bool occaNeedsSync(void *ptr);
void occaSync(void *ptr);
void occaDontSync(void *ptr);

void occaFreeUvaPtr(void *ptr);

OCCA_END_EXTERN_C

#endif
