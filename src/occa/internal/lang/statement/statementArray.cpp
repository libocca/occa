#include <occa/internal/lang/statement/statementArray.hpp>

#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    statementArray statementArray::from(statement_t &smnt) {
      statementArray arr;
      arr.push(&smnt);
      return arr;
    }

    // Flat map
    statementArray statementArray::flatMap(smntMapCallback func) const {
      statementArray arr;

      iterateStatements([&](statement_t *smnt) {
          for (statement_t *innerSmnt : func(smnt)) {
            arr.push(innerSmnt);
          }
        });

      return arr;
    }

    statementArray statementArray::flatMap(smntWithPathMapCallback func) const {
      statementArray arr;

      iterateStatements([&](statement_t *smnt, const statementArray &path) {
          for (statement_t *innerSmnt : func(smnt, path)) {
            arr.push(innerSmnt);
          }
        });

      return arr;
    }

    exprNodeArray statementArray::flatMap(smntExprMapCallback func) const {
      exprNodeArray arr;

      iterateExprNodes([&](smntExprNode smntExpr) {
          exprNode *node = func(smntExpr);
          arr.push({smntExpr.smnt, node});
        });

      return arr;
    }

    // Flat filter
    statementArray statementArray::flatFilter(smntFilterCallback func) const {
      statementArray arr;

      iterateStatements([&](statement_t *smnt) {
          if (func(smnt)) {
            arr.push(smnt);
          }
        });

      return arr;
    }

    statementArray statementArray::flatFilter(smntWithPathFilterCallback func) const {
      statementArray arr;

      iterateStatements([&](statement_t *smnt, const statementArray &path) {
          if (func(smnt, path)) {
            arr.push(smnt);
          }
        });

      return arr;
    }

    exprNodeArray statementArray::flatFilter(smntExprFilterCallback func) const {
      exprNodeArray arr;

      iterateExprNodes([&](smntExprNode smntExpr) {
          if (func(smntExpr)) {
            arr.push(smntExpr);
          }
        });

      return arr;
    }

    // Nested forEach
    void statementArray::nestedForEach(smntVoidCallback func) const {
      iterateStatements([&](statement_t *smnt) {
          func(smnt);
        });
    }

    void statementArray::nestedForEach(smntWithPathVoidCallback func) const {
      iterateStatements([&](statement_t *smnt, const statementArray &path) {
          func(smnt, path);
        });
    }

    void statementArray::nestedForEach(smntExprVoidCallback func) const {
      iterateExprNodes([&](smntExprNode smntExpr) {
          func(smntExpr);
        });
    }

    // Filter helper functions
    statementArray statementArray::filterByStatementType(const int smntType) const {
      return filter([=](statement_t *smnt) {
          return smnt->type() & smntType;
        });
    }

    statementArray statementArray::filterByStatementType(const int smntType, const std::string &attr) const {
      return filter([=](statement_t *smnt) {
          return (
            (smnt->type() & smntType)
            && smnt->hasAttribute(attr)
          );
        });
    }

    statementArray statementArray::flatFilterByStatementType(const int smntType) const {
      return flatFilter([=](statement_t *smnt) {
          return smnt->type() & smntType;
        });
    }

    statementArray statementArray::flatFilterByStatementType(const int smntType, const std::string &attr) const {
      return flatFilter([=](statement_t *smnt) {
          return (
            (smnt->type() & smntType)
            && smnt->hasAttribute(attr)
          );
        });
    }

    statementArray statementArray::filterByAttribute(const std::string &attr) const {
      return filter([&](statement_t *smnt) {
          return smnt->hasAttribute(attr);
        });
    }

    statementArray statementArray::flatFilterByAttribute(const std::string &attr) const {
      return flatFilter([&](statement_t *smnt) {
          return smnt->hasAttribute(attr);
        });
    }

    exprNodeArray statementArray::flatFilterByExprType(const int allowedExprNodeType) const {
      return flatFilter([&](smntExprNode smntExpr) {
          return (bool) (smntExpr.node->type() & allowedExprNodeType);
        });
    }

    exprNodeArray statementArray::flatFilterByExprType(const int allowedExprNodeType, const std::string &attr) const {
      return flatFilter([&](smntExprNode smntExpr) {
          return (
            (smntExpr.node->type() & allowedExprNodeType)
            && smntExpr.node->hasAttribute(attr)
          );
        });
    }

    exprNodeArray statementArray::flatFilterExprNodesByTypes(const int smntType,
                                                             const int allowedExprNodeType) const {
      return flatFilter([&](smntExprNode smntExpr) {
          return (
            (smntExpr.smnt->type() & smntType)
            && (smntExpr.node->type() & allowedExprNodeType)
          );
        });
    }

    //---[ Helper Methods ]---------------------------------
    //  |---[ Iteration Methods ]---------------------------
    void statementArray::iterateStatements(smntVoidCallback func) const {
      iterateStatements([&](statement_t *smnt, const statementArray &path) {
          func(smnt);
        });
    }

    void statementArray::iterateStatements(smntWithPathVoidCallback func) const {
      statementArray path;

      for (statement_t *smnt : data) {
        iterateStatement(func, path, *smnt);
      }
    }

    void statementArray::iterateStatement(smntWithPathVoidCallback func, statementArray &path, statement_t &smnt) const {
      func(&smnt, path);

      path.push(&smnt);

      // Iterate over all inner/child statements
      for (auto innerSmnt : smnt.getInnerStatements()) {
        iterateStatement(func, path, *innerSmnt);
      }

      if (smnt.is<blockStatement>()) {
        blockStatement &blockSmnt = (blockStatement&) smnt;

        // Then iterate over children
        for (statement_t *childSmnt : blockSmnt.children) {
          iterateStatement(func, path, *childSmnt);
        }
      }

      path.pop();
    }

    void statementArray::iterateExprNodes(smntExprVoidCallback func) const {
      iterateStatements([&](statement_t *smnt) {
          smnt->getExprNodes()
              .forEach([&](smntExprNode node) {
                  func(node);
                });

        });
    }
    //  |===================================================

    //  |---[ Transformation methods ]----------------------
    statementArray statementArray::getKernelStatements() const {
      return filterByStatementType(statementType::functionDecl, "kernel");
    }

    void statementArray::forEachKernelStatement(functionDeclVoidCallback func) const {
      getKernelStatements()
          .forEach([&](statement_t *smnt) {
              functionDeclStatement &kernelSmnt = (functionDeclStatement&) *smnt;
              func(kernelSmnt);
            });
    }

    void statementArray::forEachDeclaration(declarationVoidCallback func) const {
      forEachDeclaration([&](variableDeclaration &decl, declarationStatement &declSmnt) {
          func(decl);
        });
    }

    void statementArray::forEachDeclaration(declarationWithSmntVoidCallback func) const {
      filterByStatementType(statementType::declaration)
          .forEach([&](statement_t *smnt) {
              declarationStatement &declSmnt = (declarationStatement&) *smnt;
              for (auto &decl : declSmnt.declarations) {
                func(decl, declSmnt);
              }
            });
    }

    void statementArray::nestedForEachDeclaration(declarationVoidCallback func) const {
      nestedForEachDeclaration([&](variableDeclaration &decl, declarationStatement &declSmnt) {
          func(decl);
        });
    }

    void statementArray::nestedForEachDeclaration(declarationWithSmntVoidCallback func) const {
      flatFilterByStatementType(statementType::declaration)
          .forEach([&](statement_t *smnt) {
              declarationStatement &declSmnt = (declarationStatement&) *smnt;
              for (auto &decl : declSmnt.declarations) {
                func(decl, declSmnt);
              }
            });
    }
    //  |===================================================
    //======================================================
  }
}
