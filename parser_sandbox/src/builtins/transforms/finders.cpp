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

#include "variable.hpp"
#include "builtins/transforms/finders.hpp"

namespace occa {
  namespace lang {
    namespace transforms {
      //---[ Statement ]----------------
      statementFinder::statementFinder() :
        statementTransform() {}

      void statementFinder::getStatements(statement_t &smnt,
                                          statementPtrVector &statements_) {
        statements = &statements_;
        apply(smnt);
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
      //================================

      //---[ Expr Node ]----------------
      exprNodeFinder::exprNodeFinder() {}

      void exprNodeFinder::getExprNodes(exprNode &expr,
                                        exprNodeVector &exprNodes_) {
        exprNodes = &exprNodes_;
        apply(expr);
      }

      exprNode* exprNodeFinder::transformExprNode(exprNode &expr) {
        if (matches(expr)) {
          exprNodes->push_back(&expr);
        }
        return &expr;
      }

      exprNodeTypeFinder::exprNodeTypeFinder(const int validExprNodeTypes_) {
        validExprNodeTypes = validExprNodeTypes_;
      }

      bool exprNodeTypeFinder::matches(exprNode &expr) {
        return true;
      }

      exprNodeAttrFinder::exprNodeAttrFinder(const int validExprNodeTypes_,
                                             const std::string &attr_) :
        attr(attr_) {
        validExprNodeTypes = (validExprNodeTypes_
                              & (exprNodeType::type     |
                                 exprNodeType::variable |
                                 exprNodeType::function));
      }

      bool exprNodeAttrFinder::matches(exprNode &expr) {
        const int eType = expr.type();
        attributeTokenMap *attributes;
        if (eType & exprNodeType::type) {
          attributes = &(((typeNode&) expr).value.attributes);
        } else if (eType & exprNodeType::variable) {
          attributes = &(((variableNode&) expr).value.attributes);
        } else {
          attributes = &(((functionNode&) expr).value.attributes);
        }
        attributeTokenMap::iterator it = attributes->find(attr);
        return (it != attributes->end());
      }
      //================================
    }
    //---[ Helper Methods ]-------------
    void findStatementsByAttr(const int validStatementTypes,
                              const std::string &attr,
                              statement_t &smnt,
                              statementPtrVector &statements) {

      transforms::statementAttrFinder finder(validStatementTypes, attr);
      finder.getStatements(smnt, statements);
    }

    void findExprNodesByType(const int validExprNodeTypes,
                             exprNode &expr,
                             exprNodeVector &exprNodes) {

      transforms::exprNodeTypeFinder finder(validExprNodeTypes);
      finder.getExprNodes(expr, exprNodes);
    }

    void findExprNodesByAttr(const int validExprNodeTypes,
                             const std::string &attr,
                             exprNode &expr,
                             exprNodeVector &exprNodes) {

      transforms::exprNodeAttrFinder finder(validExprNodeTypes, attr);
      finder.getExprNodes(expr, exprNodes);
    }
    //==================================
  }
}
