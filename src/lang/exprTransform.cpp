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
#include <occa/lang/exprTransform.hpp>

namespace occa {
  namespace lang {
    exprTransform::exprTransform() :
      validExprNodeTypes(0) {}

    exprNode* exprTransform::apply(exprNode &node) {
      // Apply transform to children
      exprNodeRefVector children;
      node.setChildren(children);
      const int childCount = (int) children.size();
      for (int i = 0; i < childCount; ++i) {
        exprNode *&child = *(children[i]);
        exprNode *newChild = apply(*child);
        if (!newChild) {
          return NULL;
        }
        child = newChild;
      }

      // Apply transform to self
      exprNode *newNode = &node;
      if (node.type() & validExprNodeTypes) {
        newNode = transformExprNode(node);
        if (newNode != &node) {
          delete &node;
        }
      }
      return newNode;
    }
  }
}
