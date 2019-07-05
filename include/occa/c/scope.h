#ifndef OCCA_C_SCOPE_HEADER
#define OCCA_C_SCOPE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

OCCA_LFUNC occaScope OCCA_RFUNC occaCreateScope(occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaScopeAdd(occaScope scope,
                                        const char *name,
                                        occaType value);

OCCA_LFUNC void OCCA_RFUNC occaScopeAddConst(occaScope scope,
                                             const char *name,
                                             occaType value);

OCCA_END_EXTERN_C

#endif
