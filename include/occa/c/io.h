#ifndef OCCA_C_IO_HEADER
#define OCCA_C_IO_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

typedef void (*occaIoOutputFunction_t)(const char *str);

OCCA_LFUNC void OCCA_RFUNC occaOverrideStdout(occaIoOutputFunction_t out);

OCCA_LFUNC void OCCA_RFUNC occaOverrideStderr(occaIoOutputFunction_t out);

OCCA_END_EXTERN_C

#endif
