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
#include <occa/lang/baseStatement.hpp>
#include <occa/lang/exprNode.hpp>
#include <occa/lang/keyword.hpp>
#include <occa/lang/builtins/transforms/fillExprIdentifiers.hpp>

namespace occa {
  namespace lang {
    namespace transforms {
      fillExprIdentifiers_t::fillExprIdentifiers_t(blockStatement *scopeSmnt_) :
        scopeSmnt(scopeSmnt_) {
        validExprNodeTypes = exprNodeType::identifier;
      }

      exprNode* fillExprIdentifiers_t::transformExprNode(exprNode &node) {
        if (!scopeSmnt) {
          return &node;
        }
        const std::string &name = ((identifierNode&) node).value;
        keyword_t &keyword = scopeSmnt->getScopeKeyword(name);
        const int kType = keyword.type();
        if (!(kType & (keywordType::type     |
                       keywordType::variable |
                       keywordType::function))) {
          return &node;
        }

        if (kType & keywordType::variable) {
          return new variableNode(node.token,
                                  ((variableKeyword&) keyword).variable);
        }
        if (kType & keywordType::function) {
          return new functionNode(node.token,
                                  ((functionKeyword&) keyword).function);
        }
        // keywordType::type
        return new typeNode(node.token,
                            ((typeKeyword&) keyword).type_);
      }
    }
  }
}
