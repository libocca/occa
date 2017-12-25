#include "modes/serial.hpp"

namespace occa {
  namespace lang {
    void serialBackend::backendTransform(statement &root) {
      setupKernelArgs(root);
      modifyExclusiveVariables(root);
    }

    void serialBackend::setupKernelArgs(statement &root) {
    }

    void serialBackend::modifyExclusiveVariables(statement &root) {
    }
  }
}
