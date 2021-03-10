#ifndef OCCA_INTERNAL_C_CLI_HEADER
#define OCCA_INTERNAL_C_CLI_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

occaJson occaCliParseArgs(const int argc,
                          const char **argv,
                          const char *description);

OCCA_END_EXTERN_C

#endif
