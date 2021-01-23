#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/builtins/attributes/tile.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      tile::tile() {}

      const std::string& tile::name() const {
        static std::string name_ = "tile";
        return name_;
      }

      bool tile::forStatementType(const int sType) const {
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

      bool tile::applyCodeTransformations(blockStatement &root) {
        bool success = true;

        statementArray::from(root)
            .flatFilterByStatementType(statementType::for_, "tile")
            .forEach([&](statement_t *smnt) {
                forStatement &forSmnt = (forStatement&) *smnt;

                attributeToken_t &attr = forSmnt.attributes.find("tile")->second;
                exprNode &tileSize = *(attr.args[0].expr);

                const bool printErrors = false;
                okl::oklForStatement oklForSmnt(forSmnt, "@tile", printErrors);
                if (!oklForSmnt.isValid()) {
                  success = false;
                  return;
                }

                // Create the block and inner-block for-loops
                forStatement &blockForSmnt = *(new forStatement(forSmnt.up,
                                                                forSmnt.source));
                forStatement &innerForSmnt = *(new forStatement(&blockForSmnt,
                                                                forSmnt.source));
                blockForSmnt.add(innerForSmnt);

                // Rename the block interator
                variable_t &iter = *(oklForSmnt.iterator);
                variable_t &blockIter = iter.clone();
                blockIter.setName("_occa_tiled_" + iter.name());
                blockForSmnt.addToScope(blockIter);

                setupNewForStatements(attr,
                                      oklForSmnt,
                                      blockIter,
                                      blockForSmnt, innerForSmnt);

                setupBlockForStatement(oklForSmnt,
                                       tileSize,
                                       blockIter,
                                       blockForSmnt, innerForSmnt);

                setupInnerForStatement(oklForSmnt,
                                       tileSize,
                                       blockIter,
                                       blockForSmnt, innerForSmnt);

                setupCheckStatement(attr,
                                    oklForSmnt,
                                    blockIter,
                                    blockForSmnt, innerForSmnt);

                forSmnt.replaceWith(blockForSmnt);

                // Float @outer loop up if there are nested @tile loops
                floatOuterLoopUp(blockForSmnt);

                delete &forSmnt;
              });

        return success;
      }

      void tile::setupNewForStatements(attributeToken_t &attr,
                                       okl::oklForStatement &oklForSmnt,
                                       variable_t &blockIter,
                                       forStatement &blockForSmnt,
                                       forStatement &innerForSmnt) {
        // Add @tile attributes
        const int attrArgCount = (int) attr.args.size();
        if (attrArgCount > 1) {
          attributeTokenMap &blockAttrs = attr.args[1].attributes;
          blockForSmnt.attributes.insert(blockAttrs.begin(), blockAttrs.end());
          if (attrArgCount > 2) {
            attributeTokenMap &innerAttrs = attr.args[2].attributes;
            innerForSmnt.attributes.insert(innerAttrs.begin(), innerAttrs.end());
          }
        }
        // Remove @tile to prevent recursive updates
        innerForSmnt.attributes.erase("tile");

        forStatement &forSmnt = oklForSmnt.forSmnt;
        innerForSmnt.swap(forSmnt);

        blockForSmnt.setLoopStatements(forSmnt.init, forSmnt.check, NULL);
        innerForSmnt.setLoopStatements(NULL, NULL, forSmnt.update);
        forSmnt.setLoopStatements(NULL, NULL, NULL);

        // Replace instances of x with _occa_tiled_x
        blockForSmnt.init->replaceVariable(*oklForSmnt.iterator,
                                           blockIter);

        blockForSmnt.check->replaceVariable(*oklForSmnt.iterator,
                                            blockIter);
      }

      void tile::setupBlockForStatement(okl::oklForStatement &oklForSmnt,
                                        exprNode &tileSize,
                                        variable_t &blockIter,
                                        forStatement &blockForSmnt,
                                        forStatement &innerForSmnt) {
        /*
          for (x = START; x < END; x += INC)
          ->
          for (xTile = START; xTile < END; NULL)
          ->
          for (xTile = START; xTile < END; xTile += (TILE * (INC)))
        */
        expr innerUpdateExpr = ((expressionStatement*) innerForSmnt.update)->expr;
        expr tileSizeExpr = &tileSize;
        expr blockIterator(innerUpdateExpr.source(), blockIter);

        opType_t opType = innerUpdateExpr.opType();

        expr blockUpdate;
        if (opType & operatorType::increment) {
          //    ++IT (or IT++)
          // -> BLOCK_IT += TILE
          blockUpdate = (
            blockIterator += tileSizeExpr
          );
        }
        else if (opType & operatorType::decrement) {
          //    --IT (or IT--)
          // -> BLOCK_IT -= TILE
          blockUpdate = (
            blockIterator -= tileSizeExpr
          );
        }
        else if (opType & (operatorType::addEq | operatorType::subEq)) {
          // INC
          expr increment = innerUpdateExpr.node->to<binaryOpNode>().rightValue;

          // ((TILE) * (INC))
          expr blockIncrement = expr::parens(
            expr::parens(tileSizeExpr) * expr::parens(increment)
          );

          if (opType & operatorType::addEq) {
            blockUpdate = (
              blockIterator += blockIncrement
            );
          } else {
            blockUpdate = (
              blockIterator -= blockIncrement
            );
          }
        }

        blockForSmnt.update = blockUpdate.createStatement(&blockForSmnt, false);
      }

      void tile::setupInnerForStatement(okl::oklForStatement &oklForSmnt,
                                        exprNode &tileSize,
                                        variable_t &blockIter,
                                        forStatement &blockForSmnt,
                                        forStatement &innerForSmnt) {
        /*
          for (x = START; x < END; x += INC)
          ->
          for (NULL; NULL; x += INC)
          ->
          for (x = xTile; x < (xTile + TILE); x += INC)
        */
        auto &blockDecls = ((declarationStatement*) blockForSmnt.init)->declarations;
        token_t *declVarSource = blockDecls[0].variable().source;

        // Check variables
        expressionStatement &checkSmnt = (expressionStatement&) *blockForSmnt.check;
        binaryOpNode &checkExpr = (binaryOpNode&) *checkSmnt.expr;

        // Update variables
        expressionStatement &updateSmnt = (expressionStatement&) *blockForSmnt.update;
        binaryOpNode &updateExpr = (binaryOpNode&) *updateSmnt.expr;

        // Create init statement
        innerForSmnt.init = new declarationStatement(&innerForSmnt, declVarSource);
        auto &initDecls = ((declarationStatement*) innerForSmnt.init)->declarations;

        expr blockIterator(declVarSource, blockIter);
        expr iterator(*oklForSmnt.iterator);
        expr tileSizeExpr = &tileSize;

        initDecls.push_back(
          variableDeclaration(*oklForSmnt.iterator,
                              blockIterator.cloneExprNode())
        );

        // Create check statement
        // Note: At this point, the tile for-loop has an update
        //       with either an [+=] or [-=] update operator
        expr bounds = expr::parens(
          (updateExpr.opType() & operatorType::addEq)
          ? blockIterator + tileSizeExpr
          : blockIterator - tileSizeExpr
        );

        const binaryOperator_t &checkOp = (const binaryOperator_t&) checkExpr.op;
        expr check = (
          oklForSmnt.checkValueOnRight
          ? expr::binaryOpExpr(checkOp, iterator, bounds)
          : expr::binaryOpExpr(checkOp, bounds, iterator)
        );

        innerForSmnt.check = check.createStatement(&innerForSmnt);
      }

      void tile::setupCheckStatement(attributeToken_t &attr,
                                     okl::oklForStatement &oklForSmnt,
                                     variable_t &blockIter,
                                     forStatement &blockForSmnt,
                                     forStatement &innerForSmnt) {
        // Default to adding the check
        auto it = attr.kwargs.find("check");
        bool requiresBoundsCheck = true;
        if (it != attr.kwargs.end()) {
          requiresBoundsCheck = (bool) it->second.expr->evaluate();
        }
        if (!requiresBoundsCheck) {
          return;
        }

        // Check variables
        expressionStatement &checkSmnt = (expressionStatement&) *blockForSmnt.check;
        binaryOpNode &checkExpr = (binaryOpNode&) *checkSmnt.expr;
        token_t *checkToken = checkExpr.startNode()->token;

        // Make if statement
        ifStatement &ifSmnt = *(new ifStatement(&innerForSmnt, checkToken));
        innerForSmnt.swapChildren(ifSmnt);
        innerForSmnt.add(ifSmnt);

        expr iterator(*oklForSmnt.iterator);

        const binaryOperator_t &checkOp = (const binaryOperator_t&) checkExpr.op;
        expr check = (
          oklForSmnt.checkValueOnRight
          ? expr::binaryOpExpr(checkOp, iterator, checkExpr.rightValue)
          : expr::binaryOpExpr(checkOp, checkExpr.leftValue, iterator)
        );

        ifSmnt.setCondition(
          check.createStatement(&ifSmnt, false)
        );
      }

      void tile::floatOuterLoopUp(forStatement &outerForSmnt) {
        // TODO: This should probably be a generic @outer/@inner transformation
        //       and not just done by @tile
        if (!outerForSmnt.hasAttribute("outer")) {
          return;
        }

        // We can swap the loops as long as there is only 1 child in the parents
        // This is because @outer and @inner iterators should not be depending on
        // intermidate state that hasn't been defined
        blockStatement *newUp = outerForSmnt.up;
        while (newUp &&
               (newUp->size() == 1) &&
               !newUp->hasAttribute("outer")) {
          newUp = newUp->up;
        }

        if (!newUp ||
            (newUp == outerForSmnt.up) ||
            !newUp->hasAttribute("outer")) {
          return;
        }

        // newUp > inner1 > if > outerForSmnt > [children]
        // newUp > inner1 > outerForSmnt > if > [children]
        // newUp > outerForSmnt > inner1 > if > [children]
        blockStatement *up = outerForSmnt.up;
        while (up != newUp) {
          blockStatement *upUp = up->up;

          // State:
          //   upUp > []
          //   up > []
          //   outerForSmnt > [children]
          up->children.clear();
          upUp->children.clear();

          // State:
          //   upUp > []
          //   outerForSmnt > []
          //   up > [children]
          outerForSmnt.swapChildren(*up);

          // State:
          //   upUp > outerForSmnt > up > [children]
          upUp->add(outerForSmnt);
          outerForSmnt.add(*up);

          up = upUp;
        }
      }
    }
  }
}
