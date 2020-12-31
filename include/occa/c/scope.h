#ifndef OCCA_C_SCOPE_HEADER
#define OCCA_C_SCOPE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

occaScope occaCreateScope(occaProperties props);

void occaScopeAdd(occaScope scope,
                  const char *name,
                  occaType value);

void occaScopeAddConst(occaScope scope,
                       const char *name,
                       occaType value);

OCCA_END_EXTERN_C

#endif
