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
        attr.printError("@dim does not take kwargs");
        return false;
      }
      if (!attr.args.size()) {
        attr.printError("@dim expects at least one argument");
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
      return true;
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
      exprNodeMap::const_iterator it = attr.kwargs.begin();
      while (it != attr.kwargs.end()) {
        if (it->first != "safe") {
          it->second
            ->startNode()
            ->printError("[@tile] does not take this kwarg");
          return false;
        }
        exprNode *value = it->second;
        if (!value->canEvaluate()) {
          it->second
            ->startNode()
            ->printError("[@tile] 'safe' argument must be true or false");
          return false;
        }
        ++it;
      }
      if (!attr.args.size()) {
        attr.printError("@tile expects at least one argument");
        return false;
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
