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
#include "modes/okl.hpp"
#include "variable.hpp"
#include "builtins/types.hpp"
#include "builtins/transforms/finders.hpp"

namespace occa {
  namespace lang {
    namespace okl {
      bool checkKernels(statement_t &root) {
        // Get @kernels
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             root,
                             kernelSmnts);

        // Get @outer and @inner
        const int kernelCount = (int) kernelSmnts.size();
        if (kernelCount == 0) {
          occa::printError("No @kernels found");
          return false;
        }
        for (int i = 0; i < kernelCount; ++i) {
          if (!checkKernel(*(kernelSmnts[i]))) {
            return false;
          }
        }
        return true;
      }

      bool checkKernel(statement_t &kernelSmnt) {
        statementPtrVector outerSmnts, innerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             kernelSmnt,
                             outerSmnts);
        findStatementsByAttr(statementType::for_,
                             "inner",
                             kernelSmnt,
                             innerSmnts);
        if (!checkLoops(kernelSmnt,
                        outerSmnts, innerSmnts)) {
          return false;
        }
        // Order @outer and @inner loops
        // @outer > @shared > @inner
        // @outer > @exclusive > @inner
        // @shared has an array with evaluable sizes
        // if (!testSharedAndExclusive()) {
        //   return false;
        // }
        // No break in @outer/@inner (ok inside regular loops inside @outer/@inner)
        // No continue in @inner (ok inside regular loops inside @outer/@inner)
        // return testBreakAndContinue();
        return true;
      }

      bool checkLoops(statement_t &kernelSmnt,
                      statementPtrVector &outerSmnts,
                      statementPtrVector &innerSmnts) {
        if (!checkForDeclarations(kernelSmnt, outerSmnts, "outer")
            || !checkForDeclarations(kernelSmnt, innerSmnts, "inner")) {
          return false;
        }
        // @outer > @inner
        // Same # of @inner in each @outer
        return true;
      }

      bool checkForDeclarations(statement_t &kernelSmnt,
                                statementPtrVector &forSmnts,
                                const std::string &attrName) {
        const int count = (int) forSmnts.size();
        if (!count) {
          kernelSmnt.printError("[@kernel] requires at least one [@"
                                + attrName
                                + "] for-loop");
          return false;
        }
        for (int i = 0; i < count; ++i) {
          forStatement &forSmnt = *((forStatement*) forSmnts[i]);
          // Proper loops (decl, update, inc)
          if (!isSimpleForSmnt(attrName, forSmnt)) {
            return false;
          }
        }
        return true;
      }

      bool isSimpleForSmnt(const std::string &attrName,
                           forStatement &forSmnt) {
        variable_t *iter;
        return isSimpleForSmnt(attrName, forSmnt, iter);
      }

      bool isSimpleForSmnt(const std::string &attrName,
                           forStatement &forSmnt,
                           variable_t *&iter) {
        if (!okl::isSimpleForInit(attrName, *forSmnt.init)) {
          return false;
        }
        iter = (((declarationStatement*) forSmnt.init)
                ->declarations[0]
                .variable);
        if (!okl::isSimpleForCheck(attrName, *iter, *forSmnt.check) ||
            !okl::isSimpleForUpdate(attrName, *iter, *forSmnt.update)) {
          return false;
        }
        return true;
      }

      bool isSimpleForInit(const std::string &attrName,
                           statement_t &smnt) {
        if (smnt.type() != statementType::declaration) {
          smnt.printError("[@" + attrName + "] Expected a declaration statement");
          return false;
        }
        // Can only have one declaration
        declarationStatement &declSmnt = (declarationStatement&) smnt;
        if (declSmnt.declarations.size() > 1) {
          declSmnt.declarations[1].printError(
            "[@" + attrName + "] Can only transform 1 iterator variable"
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
          var.printError("[@" + attrName + "] Iterator variable needs to be of type"
                         " [char, short, int, long]");
          return false;
        }
        return true;
      }

      bool isSimpleForCheck(const std::string &attrName,
                            variable_t &var,
                            statement_t &smnt) {
        if (smnt.type() != statementType::expression) {
          smnt.printError("[@" + attrName + "] Expected comparing ["
                          + var.name()
                          + "] with some bound");
          return false;
        }
        // Check valid operator (<, <=, >=, >)
        exprNode &expr = *(((expressionStatement&) smnt).expr);
        if (expr.type() != exprNodeType::binary) {
          smnt.printError("[@" + attrName + "] Expected to compare ["
                          + var.name()
                          + "] with one of these operators [<, <=, >=, >]");
          return false;
        }
        binaryOpNode &opNode = (binaryOpNode&) expr;
        if (!(opNode.opType() & (operatorType::lessThan      |
                                 operatorType::lessThanEq    |
                                 operatorType::greaterThanEq |
                                 operatorType::greaterThan))) {
          smnt.printError("[@" + attrName + "] Expected to compare ["
                          + var.name()
                          + "] with one of these operators [<, <=, >=, >]");
          return false;
        }
        if (!hasSameVariable(var, opNode)) {
          smnt.printError("[@" + attrName + "] Expected to compare ["
                          + var.name()
                          + "] with one of these operators [<, <=, >=, >]");
          return false;
        }
        return true;
      }

      bool isSimpleForUpdate(const std::string &attrName,
                             variable_t &var,
                             statement_t &smnt) {
        if (smnt.type() != statementType::expression) {
          smnt.printError("[@" + attrName + "] Expected to update ["
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
          smnt.printError("[@" + attrName + "] Expected update ["
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
          validVar = hasSameVariable(var, opNode);
        }
        else if (eType == exprNodeType::rightUnary) {
          rightUnaryOpNode &opNode = (rightUnaryOpNode&) expr;
          validOp = (opNode.opType() & (operatorType::rightIncrement |
                                        operatorType::rightDecrement));
          validVar = hasSameVariable(var, opNode);
        }
        else { // eType == exprNodeType::binary
          binaryOpNode &opNode = (binaryOpNode&) expr;
          validOp = (opNode.opType() & (operatorType::addEq |
                                        operatorType::subEq));
          validVar = hasSameVariable(var, opNode);
        }
        if (!validOp) {
          expr.printError("[@" + attrName + "] Expected update ["
                          + var.name()
                          + "] with one of these operators [++, --, +=, -=]");
          return false;
        }
        if (!validVar) {
          expr.startNode()->printError("[@" + attrName + "] Expected update ["
                                       + var.name()
                                       + "] with one of these operators [++, --, +=, -=]");
          return false;
        }
        return true;
      }

      bool hasSameVariable(variable_t &var,
                           leftUnaryOpNode &opNode) {
        if (opNode.value->type() != exprNodeType::variable) {
          return false;
        }
        variable_t &var2 = ((variableNode*) opNode.value)->value;
        return (&var == &var2);
      }

      bool hasSameVariable(variable_t &var,
                           rightUnaryOpNode &opNode) {
        if (opNode.value->type() != exprNodeType::variable) {
          return false;
        }
        variable_t &var2 = ((variableNode*) opNode.value)->value;
        return (&var == &var2);
      }

      int hasSameVariable(variable_t &var,
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
    }
  }
}
