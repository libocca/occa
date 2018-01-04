#ifndef OCCA_PARSER_MODES_BACKEND_HEADER2
#define OCCA_PARSER_MODES_BACKEND_HEADER2

#include "occa/tools/properties.hpp"
#include "operator.hpp"
#include "statement.hpp"

namespace occa {
  namespace lang {
    class backend {
    public:
      const properties &props;

      backend(const properties &props_ = "");

      virtual void transform(statement &root) = 0;
    };

    class oklBackend : public backend {
    public:
      oklBackend(const properties &props_);

      virtual void transform(statement &root);
      virtual void backendTransform(statement &root) = 0;

      // @tile(...) -> for-loops
      void splitTiledOccaLoops(statement &root);

      // @outer -> @outer(#)
      void retagOccaLoops(statement &root);

      void attributeOccaLoop(forStatement &loop);

      void verifyOccaLoopInit(forStatement &loop,
                              variable *&initVar);

      void verifyOccaLoopCheck(forStatement &loop,
                               variable &initVar,
                               operator_t *&checkOp,
                               exprNode *&checkExpression);

      void verifyOccaLoopUpdate(forStatement &loop,
                                variable &initVar,
                                operator_t *&updateOp,
                                exprNode *&updateExpression);

      void splitTiledOccaLoop(forStatement &loop);

      // Check conditional barriers
      void checkOccaBarriers(statement &root);

      // Move the defines to the kernel scope
      void floatSharedAndExclusiveDefines(statement &root);
    };
  }
}

#endif
