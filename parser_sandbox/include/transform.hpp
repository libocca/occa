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

#ifndef OCCA_LANG_TRANSFORM_HEADER
#define OCCA_LANG_TRANSFORM_HEADER

namespace occa {
  namespace lang {
    class parser_t;
    class statement_t;
    class blockStatement;

    typedef statement_t* (*statementTransform_t)(statement_t &smnt);

    enum transformOrder {
      upToDown, downToUp
    };

    class statementTransform {
    public:
      parser_t &parser;
      transformOrder order;
      int validStatementTypes;

      statementTransform(parser_t &parser_);

      virtual statement_t* transformStatement(statement_t &smnt) = 0;

      statement_t* transform(statement_t &smnt);

      statement_t* transformBlockStatement(blockStatement &smnt);

      bool transformChildrenStatements(blockStatement &smnt);

      bool transformStatementInPlace(statement_t *&smnt);

      bool transformInnerStatements(blockStatement &smnt);

      bool transformForInnerStatements(forStatement &smnt);

      bool transformIfInnerStatements(ifStatement &smnt);

      bool transformElifInnerStatements(elifStatement &smnt);

      bool transformWhileInnerStatements(whileStatement &smnt);

      bool transformSwitchInnerStatements(switchStatement &smnt);
    };
  }
}

#endif
