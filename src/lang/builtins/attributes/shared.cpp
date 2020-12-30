#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/builtins/attributes/shared.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      shared::shared() {}

      const std::string& shared::name() const {
        static std::string name_ = "shared";
        return name_;
      }

      bool shared::forVariable() const {
        return true;
      }

      bool shared::forStatementType(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool shared::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size() ||
            attr.args.size()) {
          attr.printError("[@shared] does not take arguments");
          return false;
        }
        return true;
      }
    }
  }
}
