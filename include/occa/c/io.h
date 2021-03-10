#ifndef OCCA_C_IO_HEADER
#define OCCA_C_IO_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

typedef void (*occaIoOutputFunction_t)(const char *str);

void occaOverrideStdout(occaIoOutputFunction_t out);

void occaOverrideStderr(occaIoOutputFunction_t out);

OCCA_END_EXTERN_C

#endif
