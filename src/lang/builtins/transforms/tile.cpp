/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <occa/lang/exprNode.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/builtins/transforms/tile.hpp>
#include <occa/lang/builtins/transforms/replacer.hpp>

namespace occa {
  namespace lang {
    namespace transforms {
      tile::tile() {
        validStatementTypes = statementType::for_;
      }

      statement_t* tile::transformStatement(statement_t &smnt) {
        forStatement &forSmnt = (forStatement&) smnt;
        attributeTokenMap::iterator it = forSmnt.attributes.find("tile");
        if (it == forSmnt.attributes.end()) {
          return &smnt;
        }
        attributeToken_t &attr = it->second;
        exprNode &tileSize = *(attr.args[0].expr);

        okl::oklForStatement oklForSmnt(forSmnt,
                                        "@tile");
        if (!oklForSmnt.isValid()) {
          return NULL;
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
        blockForSmnt.scope.add(blockIter);

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

        return &blockForSmnt;
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

        // Setup initial statements
        blockForSmnt.setLoopStatements(forSmnt.init, forSmnt.check, NULL);
        innerForSmnt.setLoopStatements(NULL, NULL, forSmnt.update);
        forSmnt.setLoopStatements(NULL, NULL, NULL);

        // Replace instances of x with _occa_tiled_x
        replaceVariables(*blockForSmnt.init,
                         *oklForSmnt.iterator,
                         blockIter);

        replaceVariables(*blockForSmnt.check,
                         *oklForSmnt.iterator,
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
          for (xTile = START; xTile < END; NULL )
          ->
          for (xTile = START; xTile < END; xTile += (TILE * (INC)))
        */
        exprNode &updateExpr = *(((expressionStatement*) innerForSmnt.update)->expr);
        opType_t opType = ((exprOpNode&) updateExpr).opType();

        token_t *updateToken =updateExpr.startNode()->token;

        exprNode *updateSizeExpr = &tileSize;
        const binaryOperator_t *updateOp = &op::addEq;
        if (opType & (operatorType::leftDecrement |
                      operatorType::rightDecrement)) {
          updateOp = &op::subEq;
        }
        else if (opType & (operatorType::addEq |
                           operatorType::subEq)) {
          // INC
          exprNode *updateSize = ((binaryOpNode&) updateExpr).rightValue;
          // (INC)
          parenthesesNode updateInParen(updateToken,
                                        *updateSize);
          // TILE * (INC)
          binaryOpNode mult(updateToken,
                            op::mult,
                            tileSize,
                            updateInParen);
          // (TILE * (INC))
          updateSizeExpr = new parenthesesNode(updateToken,
                                               mult);
          if (opType & operatorType::subEq) {
            updateOp = &op::subEq;
          }
        }
        // VAR += (TILE * (INC))
        variableNode varNode(updateToken, blockIter);
        exprNode *newUpdateExpr = new binaryOpNode(updateToken,
                                                   *updateOp,
                                                   varNode,
                                                   *updateSizeExpr);
        if (updateSizeExpr != &tileSize) {
          // Delete (TILE * (INC)) if it was created
          delete updateSizeExpr;
        }

        blockForSmnt.update = new expressionStatement(&blockForSmnt,
                                                      *newUpdateExpr,
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
        token_t *initToken = decl.variable->source;
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
        innerForSmnt.init = new declarationStatement(&innerForSmnt);
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
        innerForSmnt.swap(ifSmnt);
        innerForSmnt.scope.swap(ifSmnt.scope);
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

      bool applyTileTransforms(statement_t &smnt) {
        tile tileTransform;
        return tileTransform.apply(smnt);
      }
    }
  }
}
