#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/attributes/simdLength.hpp>

namespace occa {
namespace lang {
namespace attributes {

  const std::string& simdLength::name() const { return name_;}

  bool simdLength::forStatementType(const int sType) const {
    return (sType & statementType::for_);
  }

  bool simdLength::isValid(const attributeToken_t &attr) const {
    if (attr.kwargs.size()) {
      attr.printError(name_ + " does not take kwargs");
      return false;
    }

    if (1 != attr.args.size()) {
      attr.printError(name_ + " takes one argument");
      return false;
    }
    
    const auto& attr_arg = attr.args[0];
    if (!attr_arg.canEvaluate()) {
      attr.printError(name_ + " cannot evaluate argument");
      return false;
    }

    primitive value = attr_arg.expr->evaluate();
    if (!value.isInteger()) {
      attr.printError(name_ + " take an integer argument");
      return false;
    }

    if(0 > value.to<int>())  {
      attr.printError(name_ + " arguments must be postive!");
      return false;
    }

    return true;
  }

}
}
}
