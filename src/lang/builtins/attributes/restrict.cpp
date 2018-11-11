#include <occa/lang/statement.hpp>
#include <occa/lang/builtins/attributes/restrict.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      restrict::restrict() {}

      std::string restrict::name() const {
        return "restrict";
      }

      bool restrict::forVariable() const {
        return true;
      }

      bool restrict::forStatement(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool restrict::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size() ||
            attr.args.size()) {
          attr.printError("[@restrict] does not take arguments");
          return false;
        }
        return true;
      }
    }
  }
}
