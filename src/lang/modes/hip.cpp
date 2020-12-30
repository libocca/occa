#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/modes/hip.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      hipParser::hipParser(const occa::properties &settings_) :
        cudaParser(settings_) {}

      void hipParser::beforeKernelSplit() {
        root.addFirst(
          *(new directiveStatement(
              &root,
              directiveToken(root.source->origin,
                             "include <hip/hip_runtime.h>")
            ))
        );
        cudaParser::beforeKernelSplit();
      }
    }
  }
}
