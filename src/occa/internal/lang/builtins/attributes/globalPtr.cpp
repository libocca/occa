#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/builtins/attributes/globalPtr.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      //---[ @globalPtr ]-----------------------
      globalPtr::globalPtr() {}

      const std::string& globalPtr::name() const {
        static std::string name_ = "globalPtr";
        return name_;
      }

      bool globalPtr::forVariable() const {
        return true;
      }

      bool globalPtr::forStatementType(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool globalPtr::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@globalPtr] does not take kwargs");
          return false;
        }
        if (attr.args.size()) {
          attr.printError("[@globalPtr] does not take arguments");
          return false;
        }
        return true;
      }
      //==================================
    }
  }
}
