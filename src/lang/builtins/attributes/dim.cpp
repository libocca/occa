#include <occa/lang/expr.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/builtins/attributes/dim.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      //---[ @dim ]-----------------------
      dim::dim() {}

      std::string dim::name() const {
        return "dim";
      }

      bool dim::forVariable() const {
        return true;
      }

      bool dim::forStatement(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool dim::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@dim] does not take kwargs");
          return false;
        }
        if (!attr.args.size()) {
          attr.printError("[@dim] expects at least one argument");
          return false;
        }
        return true;
      }
      //==================================

      //---[ @dimOrder ]------------------
      dimOrder::dimOrder() {}

      std::string dimOrder::name() const {
        return "dimOrder";
      }

      bool dimOrder::forVariable() const {
        return true;
      }

      bool dimOrder::forStatement(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool dimOrder::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@dimOrder] does not take kwargs");
          return false;
        }
        const int argCount = (int) attr.args.size();
        if (!argCount) {
          attr.printError("[@dimOrder] expects at least one argument");
          return false;
        }
        // Test valid numbers
        int *order = new int[argCount];
        ::memset(order, 0, argCount * sizeof(int));
        for (int i = 0; i < argCount; ++i) {
          // Test arg value
          exprNode *expr = attr.args[i].expr;
          if (!expr
              || !expr->canEvaluate()) {
            if (expr
                && (expr->type() != exprNodeType::empty)) {
              expr->startNode()->printError(inRangeMessage(argCount));
            } else {
              attr.printError(inRangeMessage(argCount));
            }
            delete [] order;
            return false;
          }
          // Test proper arg value
          const int i2 = (int) expr->evaluate();
          if ((i2 < 0) || (argCount <= i2)) {
            expr->startNode()->printError(inRangeMessage(argCount));
            delete [] order;
            return false;
          }
          if (order[i2]) {
            expr->startNode()->printError("[@dimOrder] Duplicate index");
            delete [] order;
            return false;
          }
          order[i2] = 1;
        }
        delete [] order;
        return true;
      }

      std::string dimOrder::inRangeMessage(const int count) const {
        std::string message = (
          "[@dimOrder] arguments must be known at compile-time"
          " and an ordering of ["
        );
        for (int i = 0; i < count; ++i) {
          if (i) {
            message += ", ";
          }
          message += occa::toString(i);
        }
        message += ']';
        return message;
      }
      //==================================
    }
  }
}
