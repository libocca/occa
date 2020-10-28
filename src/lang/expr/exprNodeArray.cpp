#include <occa/lang/statement/statementArray.hpp>

#include <occa/lang/expr.hpp>
#include <occa/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    exprNodeArray exprNodeArray::from(statement_t *smnt, exprNode *node) {
      exprNodeArray arr;

      arr.push({smnt, node});

      return arr;
    }

    void exprNodeArray::inplaceMap(smntExprMapCallback func) const {
      forEachWithRef([&](smntExprNode smntExpr, exprNode **nodeRef) {
          exprNode *mappedNode = func(smntExpr);

          if (mappedNode == smntExpr.node) {
            return;
          }

          if (smntExpr.node == smntExpr.rootNode) {
            smntExpr.smnt->replaceExprNode(smntExpr.node, mappedNode);
          } else {
            // Needs to change at the exprNode level
            delete smntExpr.node;
            *nodeRef = mappedNode;
          }
        });
    }

    void exprNodeArray::flatInplaceMap(smntExprMapCallback func) const {
      nestedForEachWithRef([&](smntExprNode smntExpr, exprNode **nodeRef) {
          exprNode *mappedNode = func(smntExpr);

          if (mappedNode == smntExpr.node) {
            return;
          }

          if (smntExpr.node == smntExpr.rootNode) {
            // Needs to change at the statement level
            smntExpr.smnt->replaceExprNode(smntExpr.node, mappedNode);
          } else {
            // Needs to change at the exprNode level
            delete smntExpr.node;
            *nodeRef = mappedNode;
          }
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

    void exprNodeArray::nestedForEach(smntExprVoidCallback func) const {
      nestedForEachWithRef([&](smntExprNode node, exprNode **nodeRef) {
          func(node);
        });
    }

    void exprNodeArray::forEachWithRef(smntExprWithRefVoidCallback func) const {
      for (auto &smntExpr : data) {
        statement_t *smnt = smntExpr.smnt;
        exprNode *node = smntExpr.node;

        // Apply transform node
        if (node) {
          func({smnt, node, node}, NULL);
        }
      }
    }

    void exprNodeArray::nestedForEachWithRef(smntExprWithRefVoidCallback func) const {
      for (auto &smntExpr : data) {
        statement_t *smnt = smntExpr.smnt;
        exprNode *node = smntExpr.node;

        if (!node) {
          continue;
        }

        // Apply transform to the node children
        exprNodeRefVector children;
        node->pushChildNodes(children);

        for (exprNode **child : children) {
          func({smnt, *child, node}, child);
        }

        // Apply transform node
        func({smnt, node, node}, NULL);
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
