#include <occa/experimental/functional/function.hpp>
#include <occa/internal/experimental/functional/functionStore.hpp>

namespace occa {
  baseFunction::baseFunction(const occa::scope &scope_) :
    scope(scope_) {}

  functionDefinition& baseFunction::definition() {
    // Should be initialized at this point
    return *functionStore.get(hash_);
  }

  hash_t baseFunction::hash() const {
    return hash_;
  }

  baseFunction::operator hash_t () const {
    return hash_;
  }
}
