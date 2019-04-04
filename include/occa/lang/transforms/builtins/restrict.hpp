#ifndef OCCA_LANG_TRANSFORMS_BUILTINS_RESTRICT_HEADER
#define OCCA_LANG_TRANSFORMS_BUILTINS_RESTRICT_HEADER

#include <occa/lang/transforms/statementTransform.hpp>

namespace occa {
  namespace lang {
    class qualifier_t;

    namespace transforms {
      class restrict : public statementTransform {
      public:
        const qualifier_t &restrictQualifier;

        restrict(const qualifier_t &restrictQualifier_);

        virtual statement_t* transformStatement(statement_t &smnt);
      };

      bool applyRestrictTransforms(statement_t &smnt,
                                   const qualifier_t &restrictQualifier);
    }
  }
}

#endif
