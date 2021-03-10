#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/attributes/exclusive.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      exclusive::exclusive() {}

      const std::string& exclusive::name() const {
        static std::string name_ = "exclusive";
        return name_;
      }

      bool exclusive::forVariable() const {
        return true;
      }

      bool exclusive::forStatementType(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool exclusive::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size() ||
            attr.args.size()) {
          attr.printError("[@exclusive] does not take arguments");
          return false;
        }
        return true;
      }
    }
  }
}
