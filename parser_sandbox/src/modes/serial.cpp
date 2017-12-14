#if 0
#include "modes/serial.hpp"

namespace occa {
  namespace lang {
    void serialBackend::transform(statement_t &root,
                                  const properties &props) {
      context_t &context = root.context;

      statementPtrVector exclusiveStatements =
        (context
         .findStatements(statementType::declaration)
         .hasAttribute("exclusive"));



      const int variableCount = (int) variables.size();
      for (int i = 0; i < variableCount ++i) {
        variable &var = *(variables[i]);
        variable.addArray();
        expressionPtrVector = context.findExpressions(var);
      }
    }
  }
}

#endif
