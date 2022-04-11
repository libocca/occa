#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/attributes/maxInnerDims.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      maxInnerDims::maxInnerDims() {}

      const std::string& maxInnerDims::name() const {
        static const std::string name_{"max_inner_dims"};
        return name_;
      }

      bool maxInnerDims::forStatementType(const int sType) const {
        return (sType & statementType::for_);
      }
      
      bool maxInnerDims::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@max_inner_dims] does not take kwargs");
          return false;
        }
        const auto argCount{attr.args.size()};
        if (1 > argCount) {
          attr.printError("[@max_inner_dims] expects at least one argument");
          return false;
        }
        if(3 < argCount) {
          attr.printError("[@max_inner_dims] takes at most 3 arguments");
          return false;
        }

        for(auto&& arg : attr.args) {
          exprNode *expr = arg.expr;
          bool error = !(expr && expr->canEvaluate());
          if(!error){
            primitive value = expr->evaluate();
            error = !value.isInteger();
            if(!error) {
              int intValue = value;
              error = !(0 < intValue);
            }
          }
          if(error) {
            attr.printError("[@max_inner_dims] arguments must be postive!");
            return false;
          }
        }
        return true;
      }
    }
  }
}
