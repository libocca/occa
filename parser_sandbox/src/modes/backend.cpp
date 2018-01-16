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
#include "modes/backend.hpp"

namespace occa {
  namespace lang {
    backend::backend(const properties &props_) :
      props(props_) {}

    void oklBackend::transform(statement_t &root) {
      // @tile(...) -> for-loops
      splitTiledOccaLoops(root);

      // @outer -> @outer(#)
      retagOccaLoops(root);

      // Check conditional barriers
      checkOccaBarriers(root);

      // Move the defines to the root scope
      floatSharedAndExclusiveDefines(root);

      backendTransform(root);
    }

    // @tile(root) -> for-loops
    void oklBackend::splitTiledOccaLoops(statement_t &root) {
#if 0
      statementQuery tiledQuery = (query::findForLoops()
                                   .withAttribute("tile"));

      statementPtrVector tiledLoops = tiledQuery(root);
      const int loopCount = (int) tiledLoops.size();
      for (int i = 0; i < loopCount; ++i) {
        forStatement &loop = tiledLoops[i]->to<forStatement>();
        tileAttribute &tile = loop.getAttribute<tileAttribute>("tile");

        attributeOccaLoop(loop);
        if (loop.hasAttribute(oklLoopInfo)) {
          splitTiledOccaLoop(loop);
        }
      }
#endif
    }

    // @outer -> @outer(#)
    void oklBackend::retagOccaLoops(statement_t &root) {
#if 0
      statementQuery outerLoops = (query::findForLoops(query::first)
                                   .withAttribute("outer"));
      statementPtrVector loops = outerLoops(kernel);
      statementPtrVector nextLoops;

      while (loops.size()) {
        const int loopCount = (int) loops.size();
        for (int i = 0; i < loopCount; ++i) {
          nextLoops = outerLoops(*(loops[i]))
        }
      }
#endif
    }

    void oklBackend::attributeOccaLoop(forStatement &loop) {
#if 0
      int errors = 0;
      if (loop.init.type() != statementType::expression) {
        loop.init.error("@outer, @inner, and @tile loops must have a simple"
                        " variable declaration statement"
                        " (e.g. for(int iter = N; ...; ...))");
        ++errors;
      }
      if (loop.check.type() != statementType::expression) {
        loop.check.error("@outer, @inner, and @tile loops must have a simple"
                         " check statement"
                         " (e.g. for(...; iter < N or N < iter; ...))");
        ++errors;
      }
      if (loop.update.type() != statementType::expression) {
        loop.update.error("@outer, @inner, and @tile loops must have a simple"
                          " update statement"
                          " (e.g. for(...; ...; ++iter or iter += N))");
        ++errors;
      }
      if (errors) {
        return;
      }

      variable *initVar;
      operator_t *checkOp, *updateOp;
      exprNode *checkExpression, *updateExpression;
      verifyOccaLoopInit(loop, initVar);
      if (!initVar) {
        return;
      }
      verifyOccaLoopCheck(loop, *initVar, checkOp, checkExpression);
      if (!checkOp || !checkExpression) {
        return;
      }
      verifyOccaLoopUpdate(loop, *initVar, updateOp, updateExpression);
      if (updateExpression) {
        loop.addAttribute(
          oklLoopInfo(initVar,
                      checkOp, checkExpression,
                      updateOp, updateExpression)
        );
      }
#endif
    }

    void verifyOccaLoopInit(forStatement &loop,
                            variable *&initVar) {
#if 0
      expressionStatement &init = loop.init.to<expressionStatement>();
      type_t *initType;
      exprNode *initExpression;
      if (!query(init.expression)
          .hasFormat(
            query::optional(
              query::type(initType)
            )
            + query::variable(initVar)
            + query::op(op::equals)
            + query::expression(initExpression)
          )) {
        loop.init.error("@outer, @inner, and @tile loops must have a simple"
                        " variable declaration statement"
                        " (e.g. for(int iter = N; ...; ...))");
        return;
      }
      if (!errors
          && !initVar->type->canBeCastedToImplicitly(int_)) {
        loop.init.error("@outer, @inner, and @tile loops iterations"
                        " must use int or long iteration indices");
      }
#endif
    }

    void verifyOccaLoopCheck(forStatement &loop,
                             variable &initVar,
                             operator_t *&checkOp,
                             exprNode *&checkExpression) {
#if 0
      expressionStatement &check = loop.check.to<expressionStatement>();
      if (!query(init.expression)
          .hasFormat(
            + query::variable(initVar)
            + query::any(
              query::op(checkOp, op::lessThan),
              query::op(checkOp, op::lessThanEq),
              query::op(checkOp, op::greaterThan),
              query::op(checkOp, op::greaterThanEq)
            )
            + query::expression(checkExpression)
          )) {
        loop.check.error("@outer, @inner, and @tile loops must have a simple"
                         " check statement"
                         " (e.g. for(...; iter < N or N < iter; ...))");
      }
#endif
    }

    void verifyOccaLoopUpdate(forStatement &loop,
                              variable &initVar,
                              operator_t *&updateOp,
                              exprNode *&updateExpression) {
#if 0
      expressionStatement &update = loop.update.to<expressionStatement>();
      errors = 0;
      if (!query(init.update)
          .hasFormat(
            + query::variable(initVar)
            + query::any(
              query::op(updateOp, op::rightIncrement),
              query::op(updateOp, op::addEq),
              query::op(updateOp, op::subEq)
            )
            + query::optional(
              query::expression(updateExpression)
            )
          )) {
        loop.update.error("@outer, @inner, and @tile loops must have a simple"
                          " update statement"
                          " (e.g. for(...; ...; ++iter or iter += N))");
        return;
      }
#endif
    }

    void oklBackend::splitTiledOccaLoop(forStatement &loop) {
#if 0
      oklLoopInfo &info = loop.getAttribute("oklLoopInfo").to<oklLoopInfo>();
      // for (int i = INIT; i OP CHECK; i OP UPDATE) {}
      // Copy:
      //   for (int i2 = INIT; i2 OP CHECK   ; i2 OP UPDATE) {}
      forStatement &outerLoop = loop.clone().to<forStatement>();
      // Change the update value:
      //   for (int i2 = INIT; i2 OP CHECK   ; i2 OP (UPDATE * TILESIZE)) {}
      variable &outerInitVar = info.initVar.clone();
      outerLoop.initVar = &outerInitVar;
      // Update the init variable
      //   for (int i = i2   ; i OP CHECK    ; i OP UPDATE) {}
      // Update the check value
      //   for (int i = i2   ; i OP TILESIZE ; i OP UPDATE) {}
#endif
    }

    // Check conditional barriers
    void oklBackend::checkOccaBarriers(statement_t &root) {
#if 0
      // Check for outer-most inner loops
      // Place after [:-1] ([:] if in a loop)
#endif
    }

      // Move the defines to the kernel scope
    void oklBackend::floatSharedAndExclusiveDefines(statement_t &root) {
#if 0
      // Move
#endif
    }
  }
}
