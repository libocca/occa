#include <occa/internal/c/types.hpp>
#include <occa/c/scope.h>

namespace occa {
  static void addToScope(occaScope scope,
                         const char *name,
                         occaType value,
                         const bool isConst) {
    occa::scope& scope_ = occa::c::scope(scope);

    const bool isPointer = (
      (value.type == OCCA_PTR) || (value.type == OCCA_MEMORY)
    );

    scope_.add(
      scopeVariable(
        occa::c::getDtype(value),
        isPointer,
        isConst,
        name,
        occa::c::kernelArg(value)
      )
    );
  }
}


OCCA_START_EXTERN_C

occaScope occaCreateScope(occaProperties props) {
  occa::scope *scope;
  if (occa::c::isDefault(props)) {
    scope = new occa::scope();
  } else {
    scope = new occa::scope(occa::c::properties(props));
  }
  return occa::c::newOccaType(*scope);
}

void occaScopeAdd(occaScope scope,
                  const char *name,
                  occaType value) {
  occa::addToScope(scope, name, value, false);
}

void occaScopeAddConst(occaScope scope,
                       const char *name,
                       occaType value) {
  occa::addToScope(scope, name, value, true);
}

OCCA_END_EXTERN_C
