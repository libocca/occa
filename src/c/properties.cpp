#include <occa/internal/c/types.hpp>
#include <occa/c/properties.h>

OCCA_START_EXTERN_C

occaType occaCreateProperties() {
  return occa::c::newOccaType(*(new occa::properties()),
                              true);
}

occaType occaCreatePropertiesFromString(const char *c) {
  return occa::c::newOccaType(*(new occa::properties(c)),
                              true);
}

occaType occaPropertiesGet(occaProperties props,
                           const char *key,
                           occaType defaultValue) {
  occa::properties& props_ = occa::c::properties(props);
  if (props_.has(key)) {
    return occa::c::newOccaType(props_[key], false);
  }
  return defaultValue;
}

void occaPropertiesSet(occaProperties props,
                       const char *key,
                       occaType value) {
  occa::properties& props_ = occa::c::properties(props);
  props_[key] = occa::c::inferJson(value);
}

bool occaPropertiesHas(occaProperties props,
                       const char *key) {
  occa::properties& props_ = occa::c::properties(props);
  return props_.has(key);
}

OCCA_END_EXTERN_C
