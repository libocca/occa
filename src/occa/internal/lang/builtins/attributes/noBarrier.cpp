#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/builtins/attributes/noBarrier.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      //---[ @nobarrier ]-----------------------
      noBarrier::noBarrier() {}

      const std::string& noBarrier::name() const {
        static std::string name_ = "nobarrier";
        return name_;
      }

      bool noBarrier::forStatementType(const int sType) const {
        return (sType & statementType::for_);
      }

      bool noBarrier::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@nobarrier] does not take kwargs");
          return false;
        }
        if (attr.args.size()) {
          attr.printError("[@nobarrier] does not take arguments");
          return false;
        }
        return true;
      }
      //==================================
    }
  }
}
