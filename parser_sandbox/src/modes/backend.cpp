#include "modes/backend.hpp"

namespace occa {
  namespace lang {
    backend::backend(const properties &props_) :
      props(props_) {}

    void oklBackend::transform(statement &root) {
      // Apply loop reordering
      reorderLoops(root);

      // @outer -> @outer(#)
      retagOccaLoops(root);

      // @tile(root) -> for-loops
      splitTileOccaFors(root);

      // Store inner/outer + dim attributes
      storeOccaInfo(root);

      // Check conditional barriers
      checkOccaBarriers(root);

      // Add barriers between for-loops
      //   that use shared memory
      addOccaBarriers(root);

      // Move the defines to the kernel scope
      floatSharedAndExclusiveDefines(root);

      backendTransform(root);
    }

    void oklBackend::reorderLoops(statement &root) {
    }

    void oklBackend::retagOccaLoops(statement &root) {
    }

    void oklBackend::splitTileOccaFors(statement &root) {
    }

    void oklBackend::storeOccaInfo(statement &root) {
    }

    void oklBackend::checkOccaBarriers(statement &root) {
    }

    void oklBackend::addOccaBarriers(statement &root) {
    }

    void oklBackend::floatSharedAndExclusiveDefines(statement &root) {
    }
  }
}
