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
#include <occa/lang/builtins/attributes/tile.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      tile::tile() {}

      std::string tile::name() const {
        return "tile";
      }

      bool tile::forStatement(const int sType) const {
        return (sType & statementType::for_);
      }

      bool tile::isValid(const attributeToken_t &attr) const {
        return (validArgs(attr)
                && validKwargs(attr));
      }

      bool tile::validArgs(const attributeToken_t &attr) const {
        const int argCount = (int) attr.args.size();
        if (!argCount) {
          attr.printError("[@tile] expects at least one argument");
          return false;
        }
        if (argCount > 3) {
          attr.printError("[@tile] takes 1-3 arguments, the last 2 being attributes"
                          " for the block and in-block loops respectively");
          return false;
        }
        if (attr.args[0].expr->type() == exprNodeType::empty) {
          attr.printError("[@tile] expects a non-empty first argument");
          return false;
        }
        for (int i = 1; i < argCount; ++i) {
          if (attr.args[i].expr->type() != exprNodeType::empty) {
            attr.args[i]
              .expr
              ->startNode()
              ->printError("[@tile] can only take attributes for the 2nd and 3rd arguments");
            return false;
          }
        }
        return true;
      }

      bool tile::validKwargs(const attributeToken_t &attr) const {
        attributeArgMap::const_iterator it = attr.kwargs.begin();
        while (it != attr.kwargs.end()) {
          if (it->first != "check") {
            it->second
              .expr
              ->startNode()
              ->printError("[@tile] does not take this kwarg");
            return false;
          }
          exprNode *value = it->second.expr;
          if (!value->canEvaluate()) {
            it->second
              .expr
              ->startNode()
              ->printError("[@tile] 'check' argument must be true or false");
            return false;
          }
          ++it;
        }
        return true;
      }
    }
  }
}
