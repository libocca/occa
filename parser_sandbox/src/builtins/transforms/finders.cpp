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

#include "statement.hpp"
#include "builtins/transforms/finders.hpp"

namespace occa {
  namespace lang {
    namespace transforms {
      //---[ Statement ]----------------
      statementFinder::statementFinder() :
        statementTransform() {}

      void statementFinder::getStatements(statementPtrVector &statements_) {
        statements = &statements_;
      }

      statement_t* statementFinder::transformStatement(statement_t &smnt) {
        if (matches(smnt)) {
          statements->push_back(&smnt);
        }
        return &smnt;
      }

      statementAttrFinder::statementAttrFinder(const int validStatementTypes_,
                                               const std::string &attr_) :
        attr(attr_) {
        validStatementTypes = validStatementTypes_;
      }

      bool statementAttrFinder::matches(statement_t &smnt) {
        attributeTokenMap::iterator it = smnt.attributes.find(attr);
        return (it != smnt.attributes.end());
      }

      void findStatements(const int validStatementTypes,
                          const std::string &attr,
                          statementPtrVector &statements) {

        statementAttrFinder finder(validStatementTypes, attr);
        finder.getStatements(statements);
      }
      //================================

      //---[ Expr Node ]----------------
      exprNodeFinder::exprNodeFinder() {}

      void exprNodeFinder::getExprNodes(exprNodeVector &exprNodes_) {
        exprNodes = &exprNodes_;
      }

      exprNode* exprNodeFinder::transformExprNode(exprNode &expr) {
        if (matches(expr)) {
          exprNodes->push_back(&expr);
        }
        return &expr;
      }
      //================================
    }
  }
}
