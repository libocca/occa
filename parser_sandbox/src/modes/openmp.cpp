#include "modes/openmp.hpp"

namespace occa {
  namespace lang {
    void openmpBackend::backendTransform(statement &root) {
      serialBackend::backendTransform(root);
      addPragmas(root);
    }
    void openmpBackend::addPragmas(statement &root) {
    }
  }
}
