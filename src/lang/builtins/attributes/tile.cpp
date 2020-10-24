#include <occa/lang/expr.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/builtins/attributes/tile.hpp>

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
                blockIter.name() = "_occa_tiled_" + iter.name();
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
        expr innerUpdateExpr = ((expressionStatement*) innerForSmnt.update)->expr;
        expr tileSizeExpr = &tileSize;
        expr blockIterator(innerUpdateExpr.source(), blockIter);

        opType_t opType = innerUpdateExpr.opType();

        expr blockUpdate;
        if (opType & (operatorType::leftIncrement | operatorType::rightIncrement)) {
          //    ++IT (or IT++)
          // -> BLOCK_IT += TILE
          blockUpdate = (
            blockIterator += tileSizeExpr
          );
        }
        else if (opType & (operatorType::leftDecrement | operatorType::rightDecrement)) {
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

        blockForSmnt.update = new expressionStatement(&blockForSmnt,
                                                      *blockUpdate.popExprNode(),
                                                      false);
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
        // Init variables
        variableDeclaration &decl = (((declarationStatement*) blockForSmnt.init)
                                     ->declarations[0]);
        token_t *initToken = decl.variable().source;
        variableNode iterNode(initToken,
                              *oklForSmnt.iterator);
        variableNode blockIterNode(initToken, blockIter);

        // Check variables
        binaryOpNode &checkExpr = ((binaryOpNode&)
                                   *(((expressionStatement*) blockForSmnt.check)->expr));
        token_t *checkToken = checkExpr.startNode()->token;

        // Update variables
        const operator_t &updateOp = (
          ((binaryOpNode&)
           *(((expressionStatement*) blockForSmnt.update)->expr)
          ).op);
        const bool addUpdate = (updateOp.opType & operatorType::addEq);

        // Create init
        innerForSmnt.init = new declarationStatement(&innerForSmnt, initToken);
        variableDeclarationVector &decls = (
          ((declarationStatement*) innerForSmnt.init)
          ->declarations
        );

        decls.push_back(
          variableDeclaration(*oklForSmnt.iterator,
                              *(blockIterNode.clone()))
        );

        // Create check
        binaryOpNode checkValueNode(checkToken,
                                    addUpdate ? op::add : op::sub,
                                    blockIterNode,
                                    tileSize);
        parenthesesNode checkInParen(checkToken,
                                     checkValueNode);

        const bool varInLeft = oklForSmnt.checkValueOnRight;
        binaryOpNode &newCheckNode = *(
          new binaryOpNode(
            checkToken,
            (const binaryOperator_t&) checkExpr.op,
            varInLeft ? (exprNode&) iterNode : (exprNode&) checkInParen,
            varInLeft ? (exprNode&) checkInParen : (exprNode&) iterNode
          ));
        innerForSmnt.check = new expressionStatement(&innerForSmnt,
                                                     newCheckNode);
      }

      void tile::setupCheckStatement(attributeToken_t &attr,
                                     okl::oklForStatement &oklForSmnt,
                                     variable_t &blockIter,
                                     forStatement &blockForSmnt,
                                     forStatement &innerForSmnt) {
        attributeArgMap::iterator it = attr.kwargs.find("check");
        bool check = true;
        if (it != attr.kwargs.end()) {
          check = (bool) it->second.expr->evaluate();
        }
        if (!check) {
          return;
        }

        // Check variables
        binaryOpNode &checkExpr = ((binaryOpNode&)
                                   *(((expressionStatement*) blockForSmnt.check)->expr));
        token_t *checkToken = checkExpr.startNode()->token;
        const bool varInLeft = oklForSmnt.checkValueOnRight;

        // Make ifStatement
        ifStatement &ifSmnt = *(new ifStatement(&innerForSmnt,
                                                checkToken));
        innerForSmnt.swapChildren(ifSmnt);
        innerForSmnt.add(ifSmnt);

        // Get global check
        token_t *iterToken = (varInLeft
                              ? checkExpr.leftValue->token
                              : checkExpr.rightValue->token);
        variableNode iterNode(iterToken,
                              *oklForSmnt.iterator);
        binaryOpNode &newCheckNode = *(
          new binaryOpNode(
            checkExpr.token,
            (const binaryOperator_t&) checkExpr.op,
            varInLeft ? (exprNode&) iterNode : *(checkExpr.leftValue),
            varInLeft ? (exprNode&) *(checkExpr.rightValue) : (exprNode&) iterNode
          ));

        ifSmnt.setCondition(new expressionStatement(&ifSmnt,
                                                    newCheckNode,
                                                    false));
      }
    }
  }
}
