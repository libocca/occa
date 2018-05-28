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
#include "builtins/transforms/variableReplacer.hpp"

namespace occa {
  namespace lang {
    namespace transforms {
      variableReplacer_t::variableReplacer_t(parser_t &parser_) :
        statementTransform(parser_),
        from(NULL),
        to(NULL) {
        validStatementTypes = (statementType::declaration |
                               statementType::expression);
        validExprNodeTypes = exprNodeType::variable;
      }

      void variableReplacer_t::set(variable_t &from_,
                                   variable_t &to_) {
        from = &from_;
        to   = &to_;
      }

      statement_t* variableReplacer_t::transformStatement(statement_t &smnt) {
        // Expression
        if (smnt.type() & statementType::expression) {
          if (applyToExpr(((expressionStatement&) smnt).expr)) {
            return &smnt;
          }
          return NULL;
        }
        // Declaration
        declarationStatement &declSmnt = (declarationStatement&) smnt;
        const int declCount = (int) declSmnt.declarations.size();
        for (int i = 0; i < declCount; ++i) {
          variableDeclaration &decl = declSmnt.declarations[i];
          if (decl.variable == from) {
            decl.variable = to;
          }
          if (!applyToExpr(decl.value)) {
            return NULL;
          }
        }
        return &smnt;
      }

      exprNode* variableReplacer_t::transformExprNode(exprNode &node) {
        if (!from || !to) {
          return &node;
        }
        variable_t &var = ((variableNode&) node).value;
        if (&var != from) {
          return &node;
        }
        return new variableNode(node.token, *to);
      }

      bool variableReplacer_t::applyToExpr(exprNode *&expr) {
        if (expr == NULL) {
          return true;
        }
        expr = exprTransform::apply(*expr);
        return expr;
      }
    }
  }
}
