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
#include <occa/lang/exprNode.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/attributes/inner.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      inner::inner() {}

      std::string inner::name() const {
        return "inner";
      }

      bool inner::forStatement(const int sType) const {
        return (sType & statementType::for_);
      }

      bool inner::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@inner] does not take kwargs");
          return false;
        }
        const int argCount = (int) attr.args.size();
        if (argCount > 1) {
          attr.printError("[@inner] takes at most one index");
          return false;
        }
        if (argCount == 1) {
          exprNode *expr = attr.args[0].expr;
          bool error = (!expr || !expr->canEvaluate());
          if (!error) {
            primitive value = expr->evaluate();
            error = !value.isInteger();
            if (!error) {
              int intValue = value;
              error = (intValue < 0) || (2 < intValue);
            }
          }
          if (error) {
            attr.printError("[@inner] argument must be 0, 1, or 2");
            return false;
          }
        }
        return true;
      }
    }
  }
}
