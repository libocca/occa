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
#include "builtins/statementTransforms.hpp"

namespace occa {
  namespace lang {
    //---[ @tile ]----------------------
    tileLoopTransform::tileLoopTransform(parser_t &parser_) :
      statementTransform(parser_) {
      downToUp = false;
      validStatementTypes = statementType::for_;
    }

    statement_t* tileLoopTransform::transformStatement(statement_t &smnt) {
      return &smnt;
    }
    //==================================

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
  }
}
