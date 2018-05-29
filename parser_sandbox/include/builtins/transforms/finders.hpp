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

#ifndef OCCA_LANG_BUILTINS_TRANSFORMS_STATEMENTFINDER_HEADER
#define OCCA_LANG_BUILTINS_TRANSFORMS_STATEMENTFINDER_HEADER

#include <vector>

#include "statementTransform.hpp"
#include "exprTransform.hpp"

namespace occa {
  namespace lang {
    typedef std::vector<statement_t*> statementPtrVector;
    typedef std::vector<exprNode*>    exprNodeVector;

    namespace transforms {
      //---[ Statement ]----------------
      class statementFinder : public statementTransform {
      private:
        statementPtrVector *statements;

      public:
        statementFinder();

        void getStatements(statement_t &smnt,
                           statementPtrVector &statements_);

        virtual statement_t* transformStatement(statement_t &smnt);

        virtual bool matches(statement_t &smnt) = 0;
      };

      class statementAttrFinder : public statementFinder {
      private:
        std::string attr;

      public:
        statementAttrFinder(const int validStatementTypes_,
                            const std::string &attr_);

        virtual bool matches(statement_t &smnt);
      };

      void findStatements(const int validStatementTypes,
                          const std::string &attr,
                          statement_t &smnt,
                          statementPtrVector &statements);
      //================================

      //---[ Expr Node ]----------------
      class exprNodeFinder : public exprTransform {
      private:
        exprNodeVector *exprNodes;

      public:
        exprNodeFinder();

        void getExprNodes(exprNodeVector &exprNodes_);

        virtual exprNode* transformExprNode(exprNode &node);

        virtual bool matches(exprNode &expr) = 0;
      };
      //================================
    }
  }
}

#endif
