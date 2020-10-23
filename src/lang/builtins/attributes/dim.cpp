#include <occa/lang/expr.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/attributes/dim.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      //---[ @dim ]-----------------------
      dim::dim() {}

      const std::string& dim::name() const {
        static std::string name_ = "dim";
        return name_;
      }

      bool dim::forVariable() const {
        return true;
      }

      bool dim::forStatementType(const int sType) const {
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

      bool dim::applyCodeTransformations(blockStatement &root) {
        bool success = true;

        root.children
            .flatFilterByExprType(exprNodeType::call)
            .inplaceMap([&](smntExprNode smntNode) -> exprNode* {
                statement_t &smnt = *smntNode.smnt;
                callNode &call = (callNode&) *smntNode.node;

                // Only looking to transform variables: var() -> var[]
                if (!(call.value->type() & exprNodeType::variable)) {
                  return &call;
                }

                // Make sure the call variable has the @dim attribute
                variable_t &var = ((variableNode*) call.value)->value;
                attributeTokenMap::iterator it = var.attributes.find("dim");
                if (it == var.attributes.end()) {
                  return &call;
                }

                // Validate the dimensions
                attributeToken_t &dimAttr = it->second;
                if (!callHasValidIndices(call, dimAttr)) {
                  success = false;
                  return NULL;
                }

                // Check @dimOrder
                const int dimCount = (int) call.args.size();
                intVector order(dimCount);
                it = var.attributes.find("dimOrder");
                if (it == var.attributes.end()) {
                  for (int i = 0; i < dimCount; ++i) {
                    order[i] = i;
                  }
                } else if (!getDimOrder(dimAttr, it->second, order)) {
                  success = false;
                  return NULL;
                }

                // Expand the dimensions:
                // x
                // y + (2 * x)
                exprNode *index = call.args[order[dimCount - 1]];
                for (int i = (dimCount - 2); i >= 0; --i) {
                  const int i2 = order[i];
                  token_t *source = call.args[i2]->token;
                  exprNode *indexInParen = index->wrapInParentheses();

                  // Don't delete the initial call.args[...]
                  if (i < (dimCount - 2)) {
                    delete index;
                  }

                  exprNode *dimInParen = dimAttr.args[i2].expr->wrapInParentheses();
                  binaryOpNode mult(source,
                                    op::mult,
                                    *dimInParen,
                                    *indexInParen);
                  delete dimInParen;
                  delete indexInParen;

                  parenthesesNode multInParen(source,
                                              mult);
                  exprNode *argInParen = call.args[i2]->wrapInParentheses();

                  index = new binaryOpNode(source,
                                           op::add,
                                           *argInParen,
                                           multInParen);
                  delete argInParen;
                }

                exprNode *newValue = new subscriptNode(call.token,
                                                       *(call.value),
                                                       *index);
                // Don't delete the initial call.args[...]
                if (dimCount > 1) {
                  delete index;
                }

                smnt.updateIdentifierReferences(newValue);

                return newValue;
              });

        return success;
      }

      bool dim::getDimOrder(attributeToken_t &dimAttr,
                            attributeToken_t &dimOrderAttr,
                            intVector &order) {
        const int dimCount   = (int) dimAttr.args.size();
        const int orderCount = (int) dimOrderAttr.args.size();

        if (dimCount < orderCount) {
          dimAttr.printError("Too many dimensions, expected "
                             + occa::toString(dimCount)
                             + " argument(s)");
          return false;
        }

        if (dimCount > orderCount) {
          dimAttr.printError("Missing dimensions, expected "
                             + occa::toString(dimCount)
                             + " argument(s)");
          return false;
        }

        for (int i = 0; i < orderCount; ++i) {
          order[i] = (int) dimOrderAttr.args[i].expr->evaluate();
        }
        return true;
      }

      bool dim::callHasValidIndices(callNode &call,
                                    attributeToken_t &dimAttr) {
        const int dimCount = (int) dimAttr.args.size();
        const int argCount = (int) call.args.size();
        if (dimCount == argCount) {
          return true;
        }

        if (dimCount < argCount) {
          call.args[dimCount]->token->printError("Too many dimensions, expected "
                                                 + occa::toString(dimCount)
                                                 + " argument(s)");
        } else {
          call.value->token->printError("Missing dimensions, expected "
                                        + occa::toString(dimCount)
                                        + " argument(s)");
        }

        return false;
      }
      //==================================

      //---[ @dimOrder ]------------------
      dimOrder::dimOrder() {}

      const std::string& dimOrder::name() const {
        static std::string name_ = "dimOrder";
        return name_;
      }

      bool dimOrder::forVariable() const {
        return true;
      }

      bool dimOrder::forStatementType(const int sType) const {
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
