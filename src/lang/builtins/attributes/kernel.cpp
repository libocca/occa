#include <occa/lang/expr.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/attributes/kernel.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      kernel::kernel() {}

      const std::string& kernel::name() const {
        static std::string name_ = "kernel";
        return name_;
      }

      bool kernel::forFunction() const {
        return true;
      }

      bool kernel::forStatement(const int sType) const {
        return (sType & (statementType::function |
                         statementType::functionDecl));
      }

      bool kernel::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size() ||
            attr.args.size()) {
          attr.printError("[@kernel] does not take arguments");
          return false;
        }
        return true;
      }
    }
  }
}
