#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/builtins/attributes/implicitArg.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      implicitArg::implicitArg() {}

      const std::string& implicitArg::name() const {
        static std::string name_ = "implicitArg";
        return name_;
      }

      bool implicitArg::forVariable() const {
        return true;
      }

      bool implicitArg::forStatementType(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool implicitArg::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size() ||
            attr.args.size()) {
          attr.printError("[@implicitArg] does not take arguments");
          return false;
        }
        return true;
      }
    }
  }
}
