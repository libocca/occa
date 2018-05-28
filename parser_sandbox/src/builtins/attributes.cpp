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
#include "builtins/attributes.hpp"
#include "exprNode.hpp"
#include "parser.hpp"
#include "statement.hpp"
#include "variable.hpp"

namespace occa {
  namespace lang {
    namespace attributes {
      //---[ @dim ]-----------------------
      dim::dim() {}

      std::string dim::name() const {
        return "dim";
      }

      bool dim::forVariable() const {
        return true;
      }

      bool dim::forStatement(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool dim::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@dim] does not take kwargs");
          return false;
        }
        if (!attr.args.size()) {
          attr.printError("[@dim] expects at least one argument");
          return false;
        }
        return true;
      }
      //==================================

      //---[ @dimOrder ]------------------
      dimOrder::dimOrder() {}

      std::string dimOrder::name() const {
        return "dimOrder";
      }

      bool dimOrder::forVariable() const {
        return true;
      }

      bool dimOrder::forStatement(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool dimOrder::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size()) {
          attr.printError("[@dimOrder] does not take kwargs");
          return false;
        }
        const int argCount = (int) attr.args.size();
        if (!argCount) {
          attr.printError("[@dimOrder] expects at least one argument");
          return false;
        }
        // Test valid numbers
        int *order = new int[argCount];
        ::memset(order, 0, argCount * sizeof(int));
        for (int i = 0; i < argCount; ++i) {
          // Test arg value
          exprNode *expr = attr.args[i].expr;
          if (!expr
              || !expr->canEvaluate()) {
            if (expr
                && (expr->type() != exprNodeType::empty)) {
              expr->startNode()->printError(inRangeMessage(argCount));
            } else {
              attr.printError(inRangeMessage(argCount));
            }
            delete [] order;
            return false;
          }
          // Test proper arg value
          const int i2 = (int) expr->evaluate();
          if ((i2 < 0) || (argCount <= i2)) {
            expr->startNode()->printError(inRangeMessage(argCount));
            delete [] order;
            return false;
          }
          if (order[i2]) {
            expr->startNode()->printError("[@dimOrder] Duplicate index");
            delete [] order;
            return false;
          }
          order[i2] = 1;
        }
        delete [] order;
        return true;
      }

      std::string dimOrder::inRangeMessage(const int count) const {
        std::string message = (
          "[@dimOrder] arguments must be known at compile-time"
          " and an ordering of ["
        );
        for (int i = 0; i < count; ++i) {
          if (i) {
            message += ", ";
          }
          message += occa::toString(i);
        }
        message += ']';
        return message;
      }
      //==================================

      //---[ @tile ]----------------------
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
          if (it->first != "safe") {
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
              ->printError("[@tile] 'safe' argument must be true or false");
            return false;
          }
          ++it;
        }
        return true;
      }
      //==================================

      //---[ @kernel ]------------------
      kernel::kernel() {}

      std::string kernel::name() const {
        return "kernel";
      }

      bool kernel::forFunction() const {
        return true;
      }

      bool kernel::forStatement(const int sType) const {
        return (sType & (statementType::function |
                         statementType::functionDecl));
      }

      bool kernel::isValid(const attributeToken_t &attr) const {
        return true;
      }
      //==================================

      //---[ @outer ]---------------------
      outer::outer() {}

      std::string outer::name() const {
        return "outer";
      }

      bool outer::forStatement(const int sType) const {
        return (sType & statementType::for_);
      }

      bool outer::isValid(const attributeToken_t &attr) const {
        return true;
      }
      //==================================

      //---[ @inner ]---------------------
      inner::inner() {}

      std::string inner::name() const {
        return "inner";
      }

      bool inner::forStatement(const int sType) const {
        return (sType & statementType::for_);
      }

      bool inner::isValid(const attributeToken_t &attr) const {
        return true;
      }
      //==================================

      //---[ @shared ]---------------------
      shared::shared() {}

      std::string shared::name() const {
        return "shared";
      }

      bool shared::forVariable() const {
        return true;
      }

      bool shared::forStatement(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool shared::isValid(const attributeToken_t &attr) const {
        return true;
      }
      //==================================

      //---[ @exclusive ]---------------------
      exclusive::exclusive() {}

      std::string exclusive::name() const {
        return "exclusive";
      }

      bool exclusive::forVariable() const {
        return true;
      }

      bool exclusive::forStatement(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool exclusive::isValid(const attributeToken_t &attr) const {
        return true;
      }
      //==================================
    }
  }
}
