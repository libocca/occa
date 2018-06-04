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
#include "occa/lang/modes/oklForStatement.hpp"
#include "occa/lang/exprNode.hpp"
#include "occa/lang/statement.hpp"
#include "occa/lang/variable.hpp"
#include "occa/lang/builtins/types.hpp"

namespace occa {
  namespace lang {
    namespace okl {
      oklForStatement::oklForStatement(forStatement &forSmnt_,
                                       const std::string &source_,
                                       const bool printErrors_) :
        forSmnt(forSmnt_),
        source(source_),
        printErrors(printErrors_),
        iterator(NULL),
        initValue(NULL),
        checkOp(NULL),
        checkValue(NULL),
        updateOp(NULL),
        valid(false) {

        valid = (
          hasValidInit()
          && hasValidCheck()
          && hasValidUpdate()
        );
      }

      bool oklForStatement::isValid() {
        return valid;
      }

      bool oklForStatement::isValid(forStatement &forSmnt_,
                                    const std::string &source_,
                                    const bool printErrors_) {
        return oklForStatement(
          forSmnt_,
          source_,
          printErrors_
        ).isValid();
      }

      bool oklForStatement::hasValidInit() {
        statement_t &initSmnt = *(forSmnt.init);
        // Check for declaration
        if (initSmnt.type() != statementType::declaration) {
          if (printErrors) {
            initSmnt.printError("[" + source + "] Expected a declaration statement");
          }
          return false;
        }
        // Can only have one declaration
        declarationStatement &declSmnt = (declarationStatement&) initSmnt;
        if (declSmnt.declarations.size() > 1) {
          if (printErrors) {
            declSmnt.declarations[1].printError(
              "[" + source + "] Can only have 1 iterator variable"
            );
          }
          return false;
        }
        // Get iterator and value
        variableDeclaration &decl = declSmnt.declarations[0];
        iterator  = decl.variable;
        initValue = decl.value;
        // Valid types: {char, short, int, long}
        const type_t *type = iterator->vartype.type;
        if (!type ||
            ((*type != char_)  &&
             (*type != short_) &&
             (*type != int_))) {
          if (printErrors) {
            iterator->printError("[" + source + "] Iterator variable needs to be of type"
                                 " [char, short, int, long]");
          }
          return false;
        }
        return true;
      }

      bool oklForStatement::hasValidCheck() {
        statement_t &checkSmnt = *(forSmnt.check);
        // Check an expression statement exists
        if (checkSmnt.type() != statementType::expression) {
          if (printErrors) {
            checkSmnt.printError("[" + source + "] Expected comparing ["
                                 + iterator->name()
                                 + "] with some bound");
          }
          return false;
        }
        // Check valid operator (<, <=, >=, >)
        exprNode &expr = *(((expressionStatement&) checkSmnt).expr);
        if (expr.type() != exprNodeType::binary) {
          if (printErrors) {
            checkSmnt.printError("[" + source + "] Expected to compare ["
                                 + iterator->name()
                                 + "] with one of these operators [<, <=, >=, >]");
          }
          return false;
        }
        // Set check operator
        checkOp = (binaryOpNode*) &expr;
        if (!(checkOp->opType() & (operatorType::lessThan      |
                                   operatorType::lessThanEq    |
                                   operatorType::greaterThanEq |
                                   operatorType::greaterThan))) {
          if (printErrors) {
            checkSmnt.printError("[" + source + "] Expected to compare ["
                                 + iterator->name()
                                 + "] with one of these operators [<, <=, >=, >]");
          }
          return false;
        }
        // Set check value
        int checkOrder = usesIterator(*checkOp,
                                      checkValue);
        if (!checkOrder) {
          if (printErrors) {
            checkSmnt.printError("[" + source + "] Expected to compare ["
                                 + iterator->name()
                                 + "] with one of these operators [<, <=, >=, >]");
          }
          return false;
        }
        checkValueOnRight = (checkOrder > 0);
        return true;
      }

      bool oklForStatement::hasValidUpdate() {
        statement_t &updateSmnt = *(forSmnt.update);
        // Check an expression statement exists
        if (updateSmnt.type() != statementType::expression) {
          if (printErrors) {
            updateSmnt.printError("[" + source + "] Expected to update ["
                                  + iterator->name()
                                  + "]");
          }
          return false;
        }
        // Check valid operator (++, --, +=, -=)
        exprNode *updateExpr = ((expressionStatement&) updateSmnt).expr;
        udim_t eType = updateExpr->type();
        if (!(eType & (exprNodeType::leftUnary  |
                       exprNodeType::rightUnary |
                       exprNodeType::binary))) {
          if (printErrors) {
            updateSmnt.printError("[" + source + "] Expected update ["
                                  + iterator->name()
                                  + "] with one of these operators [++, --, +=, -=]");
          }
          return false;
        }
        // Make sure we're using the same iterator variable
        bool validOp  = false;
        bool validVar = false;
        updateOp = (exprOpNode*) updateExpr;
        if (eType == exprNodeType::leftUnary) {
          leftUnaryOpNode &opNode = (leftUnaryOpNode&) *updateOp;
          validOp = (opNode.opType() & (operatorType::leftIncrement |
                                        operatorType::leftDecrement));
          validVar = usesIterator(opNode);
        }
        else if (eType == exprNodeType::rightUnary) {
          rightUnaryOpNode &opNode = (rightUnaryOpNode&) *updateOp;
          validOp = (opNode.opType() & (operatorType::rightIncrement |
                                        operatorType::rightDecrement));
          validVar = usesIterator(opNode);
        }
        else { // eType == exprNodeType::binary
          binaryOpNode &opNode = (binaryOpNode&) *updateOp;
          validOp = (opNode.opType() & (operatorType::addEq |
                                        operatorType::subEq));
          validVar = usesIterator(opNode, updateValue);
        }
        if (!validOp) {
          if (printErrors) {
            updateOp->printError("[" + source + "] Expected update ["
                                 + iterator->name()
                                 + "] with one of these operators [++, --, +=, -=]");
          }
          return false;
        }
        if (!validVar) {
          if (printErrors) {
            updateOp->startNode()->printError("[" + source + "] Expected update ["
                                              + iterator->name()
                                              + "] with one of these operators [++, --, +=, -=]");
          }
          return false;
        }
        return true;
      }

      bool oklForStatement::usesIterator(leftUnaryOpNode &opNode) {
        if (opNode.value->type() != exprNodeType::variable) {
          return false;
        }
        variable_t &var = ((variableNode*) opNode.value)->value;
        return (&var == iterator);
      }

      bool oklForStatement::usesIterator(rightUnaryOpNode &opNode) {
        if (opNode.value->type() != exprNodeType::variable) {
          return false;
        }
        variable_t &var = ((variableNode*) opNode.value)->value;
        return (&var == iterator);
      }

      int oklForStatement::usesIterator(binaryOpNode &opNode,
                                        exprNode *&value) {

        if (opNode.leftValue->type() == exprNodeType::variable) {
          variable_t &var = ((variableNode*) opNode.leftValue)->value;
          if (&var == iterator) {
            value = opNode.rightValue;
            return -1;
          }
        }
        if (opNode.rightValue->type() == exprNodeType::variable) {
          variable_t &var = ((variableNode*) opNode.rightValue)->value;
          if (&var == iterator) {
            value = opNode.leftValue;
            return 1;
          }
        }
        return 0;
      }

      void oklForStatement::printWarning(const std::string &message) {
        forSmnt.printWarning(message);
      }

      void oklForStatement::printError(const std::string &message) {
        forSmnt.printError(message);
      }
    }
  }
}
