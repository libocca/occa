#ifndef OCCA_INTERNAL_LANG_MODES_OPENMP_HEADER
#define OCCA_INTERNAL_LANG_MODES_OPENMP_HEADER

#include <occa/internal/lang/modes/serial.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class openmpParser : public serialParser {
       public:
        openmpParser(const occa::json &settings_ = occa::json());

        virtual void afterParsing();

        void setupOmpPragmas();

        bool isOuterForLoop(statement_t *smnt);

        void setupAtomics();

        static bool transformBlockStatement(blockStatement &blockSmnt);

        static bool transformBasicExpressionStatement(expressionStatement &exprSmnt);
      };
    }
  }
}

#endif
