#include <occa/internal/lang/statement/statementArray.hpp>

#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    exprNodeArray exprNodeArray::from(statement_t *smnt, exprNode *node) {
      exprNodeArray arr;

      arr.push({smnt, node});

      return arr;
    }

    void exprNodeArray::inplaceMap(smntExprMapCallback func) const {
      forEach([&](smntExprNode smntExpr) {
          exprNode *mappedNode = func(smntExpr);

          if (mappedNode == smntExpr.node) {
            return;
          }

          smntExpr.smnt->replaceExprNode(smntExpr.node, mappedNode);
        });
    }

    void exprNodeArray::flatInplaceMap(smntExprMapCallback func) const {
      nestedForEach([&](smntExprNode smntExpr) {
          exprNode *mappedNode = func(smntExpr);

          if (mappedNode == smntExpr.node) {
            return;
          }

          smntExpr.smnt->replaceExprNode(smntExpr.node, mappedNode);
        });
    }

    exprNodeArray exprNodeArray::flatFilter(smntExprFilterCallback func) const {
      exprNodeArray arr;

      nestedForEach([&](smntExprNode smntExpr) {
          if (func(smntExpr)) {
            arr.push(smntExpr);
          }
        });

      return arr;
    }

    void exprNodeArray::forEach(smntExprVoidCallback func) const {
      for (auto &smntExpr : data) {
        // Apply transform node
        if (smntExpr.node) {
          func(smntExpr);
        }
      }
    }

    void exprNodeArray::nestedForEach(smntExprVoidCallback func) const {
      for (auto &smntExpr : data) {
        statement_t *smnt = smntExpr.smnt;
        exprNode *node = smntExpr.node;

        if (!node) {
          continue;
        }

        // Apply transform to the node children
        for (exprNode *child : node->getNestedChildren()) {
          func({smnt, child});
        }

        // Apply transform node
        func(smntExpr);
      }
    }

    exprNodeArray exprNodeArray::flatFilterByExprType(const int allowedExprNodeType) const {
      return flatFilter([&](smntExprNode smntExpr) {
          return (bool) (smntExpr.node->type() & allowedExprNodeType);
        });
    }

    exprNodeArray exprNodeArray::flatFilterByExprType(const int allowedExprNodeType, const std::string &attr) const {
      return flatFilter([&](smntExprNode smntExpr) {
          return (
            (smntExpr.node->type() & allowedExprNodeType)
            && smntExpr.node->hasAttribute(attr)
          );
        });
    }
  }
}
