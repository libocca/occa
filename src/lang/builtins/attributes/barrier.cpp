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
#include <occa/lang/builtins/attributes/barrier.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      barrier::barrier() {}

      std::string barrier::name() const {
        return "barrier";
      }

      bool barrier::forStatement(const int sType) const {
        return (sType & statementType::empty);
      }

      bool barrier::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@barrier] does not take kwargs");
          return false;
        }
        const int argCount = (int) attr.args.size();
        if (argCount > 1) {
          attr.printError("[@barrier] takes at most one argument");
          return false;
        }
        if ((argCount == 1) &&
            (!attr.args[0].expr ||
             attr.args[0].expr->type() != exprNodeType::string)) {
          attr.printError("[@barrier] must have no arguments"
                          " or have one string argument");
          return false;
        }
        return true;
      }
    }
  }
}
