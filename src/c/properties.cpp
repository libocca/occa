#include <occa/c/types.hpp>
#include <occa/c/properties.h>

OCCA_START_EXTERN_C

occaType OCCA_RFUNC occaCreateProperties() {
  return occa::c::newOccaType(*(new occa::properties()),
                              true);
}

occaType OCCA_RFUNC occaCreatePropertiesFromString(const char *c) {
  return occa::c::newOccaType(*(new occa::properties(c)),
                              true);
}

occaType OCCA_RFUNC occaPropertiesGet(occaProperties props,
                                      const char *key,
                                      occaType defaultValue) {
  occa::properties& props_ = occa::c::properties(props);
  if (props_.has(key)) {
    return occa::c::newOccaType(props_[key], false);
  }
  return defaultValue;
}

void OCCA_RFUNC occaPropertiesSet(occaProperties props,
                                  const char *key,
                                  occaType value) {
  occa::properties& props_ = occa::c::properties(props);
  props_[key] = occa::c::inferJson(value);
}

int OCCA_RFUNC occaPropertiesHas(occaProperties props,
                                  const char *key) {
  occa::properties& props_ = occa::c::properties(props);
  return props_.has(key);
}

OCCA_END_EXTERN_C
