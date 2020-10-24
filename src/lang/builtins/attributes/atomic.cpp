#include <occa/lang/expr.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/attributes/atomic.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      atomic::atomic() {}

      const std::string& atomic::name() const {
        static std::string name_ = "atomic";
        return name_;
      }

      bool atomic::forStatementType(const int sType) const {
        return (
          sType & (
            statementType::expression
            | statementType::block
          )
        );
      }

      bool atomic::isValid(const attributeToken_t &attr) const {
        if (attr.args.size() || attr.kwargs.size()) {
          attr.printError("[@atomic] does not take arguments");
          return false;
        }
        return true;
      }
    }
  }
}
