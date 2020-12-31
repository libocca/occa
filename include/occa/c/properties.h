#ifndef OCCA_C_PROPERTIES_HEADER
#define OCCA_C_PROPERTIES_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

occaProperties occaCreateProperties();

occaProperties occaCreatePropertiesFromString(const char *c);

occaType occaPropertiesGet(occaProperties props,
                           const char *key,
                           occaType defaultValue);

void occaPropertiesSet(occaProperties props,
                       const char *key,
                       occaType value);

bool occaPropertiesHas(occaProperties props,
                       const char *key);

OCCA_END_EXTERN_C

#endif
