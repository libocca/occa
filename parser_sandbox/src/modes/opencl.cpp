#include "modes/opencl.hpp"

namespace occa {
  namespace lang {
    void openclBackend::backendTransform(statement &root) {
      addFunctionPrototypes(root);
      updateConstToConstant(root);
      addOccaFors(root);
      setupKernelArgs(root);
      setupLaunchKernel(root);
    }

    void openclBackend::addFunctionPrototypes(statement &root) {
    }

    void openclBackend::updateConstToConstant(statement &root) {
    }

    void openclBackend::addOccaFors(statement &root) {
    }

    void openclBackend::setupKernelArgs(statement &root) {
    }

    void openclBackend::setupLaunchKernel(statement &root) {
    }
  }
}
