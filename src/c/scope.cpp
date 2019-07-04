#include <occa/c/types.hpp>
#include <occa/c/scope.h>

OCCA_START_EXTERN_C

occaScope OCCA_RFUNC occaCreateScope() {
  return occa::c::newOccaType(*(new occa::scope()));
}

void OCCA_RFUNC occaScopeAddConst(occaScope scope,
                                  const char *key,
                                  occaType value) {
  occa::scope& scope_ = occa::c::scope(scope);
  // TODO:
  // scope_.addConst(key, value);
}

void OCCA_RFUNC occaScopeAdd(occaScope scope,
                             const char *key,
                             occaType value) {
  occa::scope& scope_ = occa::c::scope(scope);
  // TODO:
  // scope_.add(key, value);
}

OCCA_END_EXTERN_C
