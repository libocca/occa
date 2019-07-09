#include <occa/lang/expr.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/attributes/tile.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      tile::tile() {}

      const std::string& tile::name() const {
        static std::string name_ = "tile";
        return name_;
      }

      bool tile::forStatement(const int sType) const {
        return (sType & statementType::for_);
      }

      bool tile::isValid(const attributeToken_t &attr) const {
        return (validArgs(attr)
                && validKwargs(attr));
      }

      bool tile::validArgs(const attributeToken_t &attr) const {
        const int argCount = (int) attr.args.size();
        if (!argCount) {
          attr.printError("[@tile] expects at least one argument");
          return false;
        }
        if (argCount > 3) {
          attr.printError("[@tile] takes 1-3 arguments, the last 2 being attributes"
                          " for the block and in-block loops respectively");
          return false;
        }
        if (attr.args[0].expr->type() == exprNodeType::empty) {
          attr.printError("[@tile] expects a non-empty first argument");
          return false;
        }
        for (int i = 1; i < argCount; ++i) {
          if (attr.args[i].expr->type() != exprNodeType::empty) {
            attr.args[i]
              .expr
              ->startNode()
              ->printError("[@tile] can only take attributes for the 2nd and 3rd arguments");
            return false;
          }
        }
        return true;
      }

      bool tile::validKwargs(const attributeToken_t &attr) const {
        attributeArgMap::const_iterator it = attr.kwargs.begin();
        while (it != attr.kwargs.end()) {
          if (it->first != "check") {
            it->second
              .expr
              ->startNode()
              ->printError("[@tile] does not take this kwarg");
            return false;
          }
          exprNode *value = it->second.expr;
          if (!value->canEvaluate()) {
            it->second
              .expr
              ->startNode()
              ->printError("[@tile] 'check' argument must be true or false");
            return false;
          }
          ++it;
        }
        return true;
      }
    }
  }
}
