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
#include "exprNode.hpp"
#include "statement.hpp"
#include "variable.hpp"
#include "builtins/types.hpp"
#include "builtins/transforms/tile.hpp"

namespace occa {
  namespace lang {
    namespace transforms {
      tile::tile(parser_t &parser_) :
        statementTransform(parser_),
        variableReplacer(parser_) {
        validStatementTypes = statementType::for_;
      }

      bool tile::isValidInit(statement_t &smnt) {
        if (smnt.type() != statementType::declaration) {
          smnt.printError("[@tile] Expected a declaration statement");
          return false;
        }
        // Can only have one declaration
        declarationStatement &declSmnt = (declarationStatement&) smnt;
        if (declSmnt.declarations.size() > 1) {
          declSmnt.declarations[1].printError(
            "[@tile] Can only transform 1 iterator variable"
          );
          return false;
        }
        variableDeclaration &decl = declSmnt.declarations[0];
        // Valid types: {char, short, int, long}
        variable_t &var = *decl.variable;
        const type_t *type = var.vartype.type;
        if (!type ||
            ((*type != char_)  &&
             (*type != short_) &&
             (*type != int_))) {
          var.printError("[@tile] Iterator variable needs to be of type"
                         " [char, short, int, long]");
          return false;
        }
        return true;
      }

      bool tile::isValidCheck(variable_t &var,
                              statement_t &smnt) {
        if (smnt.type() != statementType::expression) {
          smnt.printError("[@tile] Expected comparing ["
                          + var.name()
                          + "] with some bound");
          return false;
        }
        // Check valid operator (<, <=, >=, >)
        exprNode &expr = *(((expressionStatement&) smnt).expr);
        if (expr.type() != exprNodeType::binary) {
          smnt.printError("[@tile] Expected to compare ["
                          + var.name()
                          + "] with one of these operators [<, <=, >=, >]");
          return false;
        }
        binaryOpNode &opNode = (binaryOpNode&) expr;
        if (!(opNode.opType() & (operatorType::lessThan      |
                                 operatorType::lessThanEq    |
                                 operatorType::greaterThanEq |
                                 operatorType::greaterThan))) {
          smnt.printError("[@tile] Expected to compare ["
                          + var.name()
                          + "] with one of these operators [<, <=, >=, >]");
          return false;
        }
        if (!sameVariable(var, opNode)) {
          smnt.printError("[@tile] Expected to compare ["
                          + var.name()
                          + "] with one of these operators [<, <=, >=, >]");
          return false;
        }
        return true;
      }

      bool tile::isValidUpdate(variable_t &var,
                               statement_t &smnt) {
        if (smnt.type() != statementType::expression) {
          smnt.printError("[@tile] Expected to update ["
                          + var.name()
                          + "]");
          return false;
        }
        // Check valid operator (++, --, +=, -=)
        exprNode &expr = *(((expressionStatement&) smnt).expr);
        udim_t eType = expr.type();
        if (!(eType & (exprNodeType::leftUnary  |
                       exprNodeType::rightUnary |
                       exprNodeType::binary))) {
          smnt.printError("[@tile] Expected update ["
                          + var.name()
                          + "] with one of these operators [++, --, +=, -=]");
          return false;
        }
        bool validOp  = false;
        bool validVar = false;
        if (eType == exprNodeType::leftUnary) {
          leftUnaryOpNode &opNode = (leftUnaryOpNode&) expr;
          validOp = (opNode.opType() & (operatorType::leftIncrement |
                                        operatorType::leftDecrement));
          validVar = sameVariable(var, opNode);
        }
        else if (eType == exprNodeType::rightUnary) {
          rightUnaryOpNode &opNode = (rightUnaryOpNode&) expr;
          validOp = (opNode.opType() & (operatorType::rightIncrement |
                                        operatorType::rightDecrement));
          validVar = sameVariable(var, opNode);
        }
        else { // eType == exprNodeType::binary
          binaryOpNode &opNode = (binaryOpNode&) expr;
          validOp = (opNode.opType() & (operatorType::addEq |
                                        operatorType::subEq));
          validVar = sameVariable(var, opNode);
        }
        if (!validOp) {
          expr.printError("[@tile] Expected update ["
                          + var.name()
                          + "] with one of these operators [++, --, +=, -=]");
          return false;
        }
        if (!validVar) {
          expr.startNode()->printError("[@tile] Expected update ["
                                       + var.name()
                                       + "] with one of these operators [++, --, +=, -=]");
          return false;
        }
        return true;
      }

      bool tile::sameVariable(variable_t &var,
                              leftUnaryOpNode &opNode) {
        if (opNode.value->type() != exprNodeType::variable) {
          return false;
        }
        variable_t &var2 = ((variableNode*) opNode.value)->value;
        return (&var == &var2);
      }

      bool tile::sameVariable(variable_t &var,
                              rightUnaryOpNode &opNode) {
        if (opNode.value->type() != exprNodeType::variable) {
          return false;
        }
        variable_t &var2 = ((variableNode*) opNode.value)->value;
        return (&var == &var2);
      }

      int tile::sameVariable(variable_t &var,
                              binaryOpNode &opNode) {
        variable_t *checkVar = NULL;
        if (opNode.leftValue->type() == exprNodeType::variable) {
          checkVar = &(((variableNode*) opNode.leftValue)->value);
          if (checkVar && (checkVar == &var)) {
            return -1;
          }
        }
        if (opNode.rightValue->type() == exprNodeType::variable) {
          checkVar = &(((variableNode*) opNode.rightValue)->value);
          if (checkVar && (checkVar == &var)) {
            return 1;
          }
        }
        return 0;
      }

      statement_t* tile::transformStatement(statement_t &smnt) {
        forStatement &forSmnt = (forStatement&) smnt;
        attributeTokenMap::iterator it = forSmnt.attributes.find("tile");
        if (it == forSmnt.attributes.end()) {
          return &smnt;
        }
        attributeToken_t &attr = it->second;
        exprNode &tileSize = *(attr.args[0].expr);

        if (!isValidInit(*forSmnt.init)) {
          return NULL;
        }
        variable_t &iter = *(((declarationStatement*) forSmnt.init)
                             ->declarations[0]
                             .variable);
        if (!isValidCheck(iter, *forSmnt.check) ||
            !isValidUpdate(iter, *forSmnt.update)) {
          return NULL;
        }

        // Create the block and inner-block for-loops
        forStatement &blockForSmnt = *(new forStatement(forSmnt.up,
                                                        forSmnt.source->clone()));
        forStatement &innerForSmnt = *(new forStatement(&blockForSmnt,
                                                        forSmnt.source->clone()));
        blockForSmnt.add(innerForSmnt);

        // Rename the block interator
        variable_t &blockIter = iter.clone();
        blockIter.name() = "_occa_tiled_" + iter.name();

        setupNewForStatements(attr,
                              forSmnt,
                              iter, blockIter,
                              blockForSmnt, innerForSmnt);

        setupBlockForStatement(tileSize,
                               blockIter,
                               blockForSmnt, innerForSmnt);

        setupInnerForStatement(tileSize,
                               iter, blockIter,
                               blockForSmnt, innerForSmnt);

        setupSafeStatement(attr,
                           iter, blockIter,
                           blockForSmnt, innerForSmnt);

        return &blockForSmnt;
      }

      void tile::setupNewForStatements(attributeToken_t &attr,
                                       forStatement &forSmnt,
                                       variable_t &iter,
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
            blockForSmnt.attributes.insert(innerAttrs.begin(), innerAttrs.end());
          }
        }
        // Remove @tile to prevent recursive updates
        innerForSmnt.attributes.erase("tile");

        innerForSmnt.swap(forSmnt);

        // Setup initial statements
        blockForSmnt.setLoopStatements(forSmnt.init, forSmnt.check, NULL);
        innerForSmnt.setLoopStatements(NULL, NULL, forSmnt.update);
        forSmnt.setLoopStatements(NULL, NULL, NULL);

        // Replace instances of x with _occa_tiled_x
        variableReplacer.set(iter, blockIter);
        variableReplacer.statementTransform::apply(*blockForSmnt.init);
        variableReplacer.statementTransform::apply(*blockForSmnt.check);
      }

      void tile::setupBlockForStatement(exprNode &tileSize,
                                        variable_t &blockIter,
                                        forStatement &blockForSmnt,
                                        forStatement &innerForSmnt) {
        // TODO: Free tokens
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
          parenthesesNode updateInParen(updateToken->clone(),
                                        *updateSize);
          // TILE * (INC)
          binaryOpNode mult(updateToken->clone(),
                            op::mult,
                            tileSize,
                            updateInParen);
          // (TILE * (INC))
          updateSizeExpr = new parenthesesNode(updateToken->clone(),
                                               mult);
          if (opType & operatorType::subEq) {
            updateOp = &op::subEq;
          }
        }
        // VAR += (TILE * (INC))
        variableNode varNode(updateToken->clone(), blockIter);
        exprNode *newUpdateExpr = new binaryOpNode(updateToken->clone(),
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

      void tile::setupInnerForStatement(exprNode &tileSize,
                                        variable_t &iter,
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
        variableNode iterNode(initToken->clone(), iter);
        variableNode blockIterNode(initToken->clone(), blockIter);

        // Check variables
        binaryOpNode &checkExpr = ((binaryOpNode&)
                                   *(((expressionStatement*) blockForSmnt.check)->expr));
        token_t *checkToken = checkExpr.startNode()->token;
        const bool varInLeft = sameVariable(blockIter, checkExpr) < 0;

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
          variableDeclaration(iter, *(blockIterNode.clone()))
        );

        // Create check
        binaryOpNode checkValueNode(checkToken->clone(),
                                    addUpdate ? op::add : op::sub,
                                    blockIterNode,
                                    tileSize);
        parenthesesNode checkInParen(checkToken->clone(),
                                     checkValueNode);
        binaryOpNode &newCheckNode = *(
          new binaryOpNode(
            checkToken->clone(),
            (const binaryOperator_t&) checkExpr.op,
            varInLeft ? (exprNode&) iterNode : (exprNode&) checkInParen,
            varInLeft ? (exprNode&) checkInParen : (exprNode&) iterNode
          ));
        innerForSmnt.check = new expressionStatement(&innerForSmnt,
                                                     newCheckNode);
      }

      void tile::setupSafeStatement(attributeToken_t &attr,
                                    variable_t &iter,
                                    variable_t &blockIter,
                                    forStatement &blockForSmnt,
                                    forStatement &innerForSmnt) {
        attributeArgMap::iterator it = attr.kwargs.find("safe");
        if (it == attr.kwargs.end()) {
          return;
        }
        // Check if safe=true
        const bool safe = (bool) it->second.expr->evaluate();
        if (!safe) {
          return;
        }
        // Check variables
        binaryOpNode &checkExpr = ((binaryOpNode&)
                                   *(((expressionStatement*) blockForSmnt.check)->expr));
        token_t *checkToken = checkExpr.startNode()->token;
        const bool varInLeft = sameVariable(blockIter, checkExpr) < 0;
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
        variableNode iterNode(iterToken->clone(),
                              iter);
        binaryOpNode &newCheckNode = *(
          new binaryOpNode(
            checkExpr.token->clone(),
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
