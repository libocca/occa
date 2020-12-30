#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/attributes/barrier.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      barrier::barrier() {}

      const std::string& barrier::name() const {
        static std::string name_ = "barrier";
        return name_;
      }

      bool barrier::forStatementType(const int sType) const {
        return (sType & statementType::empty);
      }

      bool barrier::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@barrier] does not take kwargs");
          return false;
        }
        const int argCount = (int) attr.args.size();
        if (argCount > 1) {
          attr.printError("[@barrier] takes at most one argument");
          return false;
        }
        if ((argCount == 1) &&
            (!attr.args[0].expr ||
             attr.args[0].expr->type() != exprNodeType::string)) {
          attr.printError("[@barrier] must have no arguments"
                          " or have one string argument");
          return false;
        }
        return true;
      }
    }
  }
}
