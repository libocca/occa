#include <occa/lang/statement.hpp>
#include <occa/lang/builtins/attributes/shared.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      shared::shared() {}

      std::string shared::name() const {
        return "shared";
      }

      bool shared::forVariable() const {
        return true;
      }

      bool shared::forStatement(const int sType) const {
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
