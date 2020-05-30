#include <occa/lang/statement.hpp>
#include <occa/lang/builtins/attributes/restrict.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      occaRestrict::occaRestrict() {}

      const std::string& occaRestrict::name() const {
        static std::string name_ = "restrict";
        return name_;
      }

      bool occaRestrict::forVariable() const {
        return true;
      }

      bool occaRestrict::forStatement(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool occaRestrict::isValid(const attributeToken_t &attr) const {
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
