#ifndef OCCA_LANG_MODES_HIP_HEADER
#define OCCA_LANG_MODES_HIP_HEADER

#include <occa/lang/mode/cuda.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class hipParser : public cudaParser {
      public:
        hipParser(const occa::properties &settings_ = occa::properties());

        virtual void beforeKernelSplit();
      };
    }
  }
}

#endif
