#ifndef OCCA_EXPERIMENTAL_FUNCTIONAL_FUNCTIONSTORE_HEADER
#define OCCA_EXPERIMENTAL_FUNCTIONAL_FUNCTIONSTORE_HEADER

#include <occa/utils/hash.hpp>
#include <occa/internal/utils/store.hpp>
#include <occa/functional/functionDefinition.hpp>

namespace occa {
  extern store_t<hash_t, functionDefinition> functionStore;
}

#endif
