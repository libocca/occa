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
#ifndef OCCA_PARSER_MODES_BACKEND_HEADER2
#define OCCA_PARSER_MODES_BACKEND_HEADER2

#include "occa/tools/properties.hpp"
#include "operator.hpp"
#include "statement.hpp"

namespace occa {
  namespace lang {
    class backend {
    public:
      const properties &props;

      backend(const properties &props_ = "");

      virtual void transform(statement &root) = 0;
    };

    class oklBackend : public backend {
    public:
      oklBackend(const properties &props_);

      virtual void transform(statement &root);
      virtual void backendTransform(statement &root) = 0;

      // @tile(...) -> for-loops
      void splitTiledOccaLoops(statement &root);

      // @outer -> @outer(#)
      void retagOccaLoops(statement &root);

      void attributeOccaLoop(forStatement &loop);

      void verifyOccaLoopInit(forStatement &loop,
                              variable *&initVar);

      void verifyOccaLoopCheck(forStatement &loop,
                               variable &initVar,
                               operator_t *&checkOp,
                               exprNode *&checkExpression);

      void verifyOccaLoopUpdate(forStatement &loop,
                                variable &initVar,
                                operator_t *&updateOp,
                                exprNode *&updateExpression);

      void splitTiledOccaLoop(forStatement &loop);

      // Check conditional barriers
      void checkOccaBarriers(statement &root);

      // Move the defines to the kernel scope
      void floatSharedAndExclusiveDefines(statement &root);
    };
  }
}

#endif
