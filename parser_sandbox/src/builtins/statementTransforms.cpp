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
#include "builtins/statementTransforms.hpp"

namespace occa {
  namespace lang {
    //---[ @dim ]-----------------------
    dimArrayTransform::eT::eT(parser_t &parser_) :
      parser(parser_),
      scopeSmnt(NULL) {
      validExprNodeTypes = exprNodeType::call;
    }

    exprNode* dimArrayTransform::eT::transformExprNode(exprNode &node) {
      callNode &call = (callNode&) node;
      if (!(call.value->type() & exprNodeType::variable)) {
        return &node;
      }

      variable_t &var = ((variableNode*) call.value)->value;
      attributeTokenMap::iterator it = var.attributes.find("dim");
      if (it == var.attributes.end()) {
        return &node;
      }
      attributeToken_t &attr = it->second;

      if (!isValid(call, attr)) {
        return NULL;
      }

      // TODO: Delete token propertly
      const int dimCount = (int) call.args.size();
      exprNode *index = call.args[dimCount - 1];
      for (int i = (dimCount - 2); i >= 0; --i) {
        binaryOpNode mult(new operatorToken(fileOrigin(),
                                            op::mult),
                          op::mult,
                          *(attr.args[i]),
                          *index);
        // Don't delete the initial call.args[...]
        if (i < (dimCount - 2)) {
          delete index;
        }

        parenthesesNode paren(new operatorToken(fileOrigin(),
                                                op::parenthesesStart),
                              mult);

        index = new binaryOpNode(new operatorToken(fileOrigin(),
                                                   op::add),
                                 op::add,
                                 *(call.args[i]),
                                 paren);
      }
      exprNode *newValue = new subscriptNode(call.token,
                                             *(call.value),
                                             *index);

      // Don't delete the initial call.args[...]
      if (dimCount > 1) {
        delete index;
      }

      return newValue;
    }

    bool dimArrayTransform::eT::isValid(callNode &call,
                                        attributeToken_t &attr) {
      const int dimCount = (int) attr.args.size();
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

    dimArrayTransform::dimArrayTransform(parser_t &parser_) :
      statementTransform(parser_),
      eTransform(parser_) {
      validStatementTypes = (statementType::expression |
                             statementType::declaration);
    }

    statement_t* dimArrayTransform::transformStatement(statement_t &smnt) {
      bool success = true;
      if (smnt.type() & statementType::expression) {
        success = apply(smnt, ((expressionStatement&) smnt).root);
      } else {
        success = applyToDeclStatement((declarationStatement&) smnt);
      }

      return success ? &smnt : NULL;
    }

    bool dimArrayTransform::applyToDeclStatement(declarationStatement &smnt) {
      const int declCount = (int) smnt.declarations.size();
      for (int i = 0; i < declCount; ++i) {
        if (!apply(smnt, smnt.declarations[i].value)) {
          return false;
        }
      }
      return true;
    }

    bool dimArrayTransform::apply(statement_t &smnt,
                                  exprNode *&expr) {
      if (expr == NULL) {
        return true;
      }
      eTransform.scopeSmnt = &smnt;
      expr = eTransform.transform(*expr);
      return expr;
    }
    //==================================

    //---[ @tile ]----------------------
    tileLoopTransform::tileLoopTransform(parser_t &parser_) :
      statementTransform(parser_) {
      validStatementTypes = statementType::for_;
    }

    statement_t* tileLoopTransform::transformStatement(statement_t &smnt) {
      forStatement &forSmnt = (forStatement&) smnt;
      attributeTokenMap::iterator it = forSmnt.attributes.find("tile");
      if (it == forSmnt.attributes.end()) {
        return &smnt;
      }
      // attributeToken_t &attr = it->second;

      if (!isValidInit(*forSmnt.init)) {
        return NULL;
      }
      variable_t &var = (((declarationStatement*) forSmnt.init)
                         ->declarations[0]
                         .var);
      if (!isValidCheck(var, *forSmnt.check) ||
          !isValidUpdate(var, *forSmnt.update)) {
        return NULL;
      }

      return &smnt;
    }

    bool tileLoopTransform::isValidInit(statement_t &smnt) {
      if (smnt.type() != statementType::declaration) {
        smnt.printError("[@tile] Expected a declaration statement");
        return false;
      }
      // Can only have one declaration
      declarationStatement &declSmnt = (declarationStatement&) smnt;
      if (declSmnt.declarations.size() > 1) {
        declSmnt.declarations[1].printError("[@tile] Can only transform 1 iterator variable");
        return false;
      }
      // Valid types: {char, short, int, long}
      variable_t &var = declSmnt.declarations[0].var;
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

    bool tileLoopTransform::isValidCheck(variable_t &var,
                                         statement_t &smnt) {
      if (smnt.type() != statementType::expression) {
        smnt.printError("[@tile] Expected comparing ["
                        + var.name()
                        + "] with some bound");
        return false;
      }
      // Check valid operator (<, <=, >=, >)
      exprNode &expr = *(((expressionStatement&) smnt).root);
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

    bool tileLoopTransform::isValidUpdate(variable_t &var,
                                          statement_t &smnt) {
      if (smnt.type() != statementType::expression) {
        smnt.printError("[@tile] Expected to update ["
                        + var.name()
                        + "]");
        return false;
      }
      // Check valid operator (++, --, +=, -=)
      exprNode &expr = *(((expressionStatement&) smnt).root);
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
        expr.token->printError("[@tile] Expected update ["
                               + var.name()
                               + "] with one of these operators [++, --, +=, -=]");
        return false;
      }
      if (!validVar) {
        smnt.printError("[@tile] Expected update ["
                        + var.name()
                        + "] with one of these operators [++, --, +=, -=]");
        return false;
      }
      return true;
    }

    bool tileLoopTransform::sameVariable(variable_t &var,
                                         leftUnaryOpNode &opNode) {
      if (opNode.value->type() != exprNodeType::variable) {
        return false;
      }
      variable_t &var2 = ((variableNode*) opNode.value)->value;
      return (var.name() == var2.name());
    }

    bool tileLoopTransform::sameVariable(variable_t &var,
                                         rightUnaryOpNode &opNode) {
      if (opNode.value->type() != exprNodeType::variable) {
        return false;
      }
      variable_t &var2 = ((variableNode*) opNode.value)->value;
      return (var.name() == var2.name());
    }

    bool tileLoopTransform::sameVariable(variable_t &var,
                                         binaryOpNode &opNode) {
      variable_t *checkVar = NULL;
      if (opNode.leftValue->type() == exprNodeType::variable) {
        checkVar = &(((variableNode*) opNode.leftValue)->value);
      }
      if (opNode.rightValue->type() == exprNodeType::variable) {
        checkVar = &(((variableNode*) opNode.rightValue)->value);
      }
      // Check matching variables
      if (!checkVar ||
          (checkVar->name() != var.name())) {
        return false;
      }
      return true;
    }
    //==================================
  }
}
