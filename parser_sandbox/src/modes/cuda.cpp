#include "modes/cuda.hpp"

namespace occa {
  namespace lang {
    void cudaBackend::backendTransform(statement &root) {
      updateConstToConstant(root);
      addOccaFors(root);
      setupKernelArgs(root);
      setupLaunchKernel(root);
    }

    void cudaBackend::updateConstToConstant(statement &root) {
    }

    void cudaBackend::addOccaFors(statement &root) {
    }

    void cudaBackend::setupKernelArgs(statement &root) {
    }

    void cudaBackend::setupLaunchKernel(statement &root) {
    }
  }
}
