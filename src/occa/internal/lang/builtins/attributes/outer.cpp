#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/attributes/outer.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      outer::outer() {}

      const std::string& outer::name() const {
        static std::string name_ = "outer";
        return name_;
      }

      bool outer::forStatementType(const int sType) const {
        return (sType & statementType::for_);
      }

      bool outer::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@outer] does not take kwargs");
          return false;
        }
        const int argCount = (int) attr.args.size();
        if (argCount > 1) {
          attr.printError("[@outer] takes at most one index");
          return false;
        }
        if (argCount == 1) {
          exprNode *expr = attr.args[0].expr;
          bool error = (!expr || !expr->canEvaluate());
          if (!error) {
            primitive value = expr->evaluate();
            error = !value.isInteger();
            if (!error) {
              int intValue = value;
              error = (intValue < 0) || (2 < intValue);
            }
          }
          if (error) {
            attr.printError("[@outer] argument must be 0, 1, or 2");
            return false;
          }
        }
        return true;
      }
    }
  }
}
