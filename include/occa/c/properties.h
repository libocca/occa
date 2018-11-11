#ifndef OCCA_C_PROPERTIES_HEADER
#define OCCA_C_PROPERTIES_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

OCCA_LFUNC occaProperties OCCA_RFUNC occaCreateProperties();

OCCA_LFUNC occaProperties OCCA_RFUNC occaCreatePropertiesFromString(const char *c);

OCCA_LFUNC occaType OCCA_RFUNC occaPropertiesGet(occaProperties props,
                                                 const char *key,
                                                 occaType defaultValue);

OCCA_LFUNC void OCCA_RFUNC occaPropertiesSet(occaProperties props,
                                             const char *key,
                                             occaType value);

OCCA_LFUNC int OCCA_RFUNC occaPropertiesHas(occaProperties props,
                                            const char *key);

OCCA_END_EXTERN_C

#endif
