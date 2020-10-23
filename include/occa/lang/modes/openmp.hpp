#ifndef OCCA_LANG_MODES_OPENMP_HEADER
#define OCCA_LANG_MODES_OPENMP_HEADER

#include <occa/lang/modes/serial.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class openmpParser : public serialParser {
      public:
        openmpParser(const occa::properties &settings_ = occa::properties());

        virtual void afterParsing();

        bool isOuterForLoop(statement_t *smnt);
      };
    }
  }
}

#endif
