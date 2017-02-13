/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa/defines.hpp"

namespace occa {
  namespace parserNS {
    //---[ Node ]-----------------------------------
    template <class TM>
    node<TM>::node() :
      left(NULL),
      right(NULL),
      up(NULL),
      down(NULL),
      value() {}

    template <class TM>
    node<TM>::node(const TM &t) :
      left(NULL),
      right(NULL),
      up(NULL),
      down(NULL),

      value(t) {}

    template <class TM>
    node<TM>::node(const node<TM> &n) :
      left(n.left),
      right(n.right),
      up(n.up),
      down(n.down),

      value(n.value) {}

    template <class TM>
    node<TM>& node<TM>::operator = (const node<TM> &n) {
      left  = n.left;
      right = n.right;
      up    = n.up;
      down  = n.down;

      value = n.value;

      return *this;
    }

    template <class TM>
    node<TM>* node<TM>::pop() {
      if (left != NULL)
        left->right = right;

      if (right != NULL)
        right->left = left;

      if (up && (up->down == this))
        up->down = right;

      return this;
    }

    template<class TM>
    node<TM>* node<TM>::push(node<TM> *n) {
      // Copy up
      n->up = up;

      node *rr = right;

      right = n;

      right->left  = this;
      right->right = rr;

      if (rr)
        rr->left = right;

      return right;
    }

    template<class TM>
    node<TM>* node<TM>::push(node<TM> *nStart,
                             node<TM> *nEnd) {
      node *rr = right;

      right = nStart;

      right->left  = this;
      nEnd->right  = rr;

      if (rr)
        rr->left = nEnd;

      // Copy up
      rr = nStart;

      while(rr != nEnd) {
        rr->up = up;
        rr = rr->right;
      }

      return nEnd;
    }

    template <class TM>
    node<TM>* node<TM>::push(const TM &t) {
      return push(new node(t));
    }

    template <class TM>
    node<TM>* node<TM>::pushDown(node<TM> *n) {
      n->up = this;
      down  = n;
      return n;
    }

    template<class TM>
    node<TM>* node<TM>::pushDown(const TM &t) {
      return pushDown(new node(t));
    }

    template<class TM>
    void node<TM>::print(const std::string &tab) {
      node *nodePos = this;

      while(nodePos) {
        printf("--------------------------------------------\n");

        if (down == NULL) {
          std::cout << tab << "[" << (std::string) *nodePos << "]\n";
        } else {
          std::cout << tab << "  " << "[" << (std::string) *nodePos << "]\n";
          down->print(tab + "  ");
        }

        printf("--------------------------------------------\n");

        nodePos = nodePos->right;
      }
    }

    template <class TM>
    void node<TM>::printPtr(const std::string &tab) {
      node *nodePos = this;

      while(nodePos) {
        printf("--------------------------------------------\n");

        if (down == NULL) {
          std::cout << tab << "[" << (std::string) *(nodePos->value) << "]\n";
        } else {
          std::cout << tab << "  " << "[" << (std::string) *(nodePos->value) << "]\n";
          down->print(tab + "  ");
        }

        printf("--------------------------------------------\n");

        nodePos = nodePos->right;
      }
    }

    template <class TM>
    void popAndGoRight(node<TM> *&n) {
      node<TM> *right = n->right;

      delete n->pop();

      n = right;
    }

    template <class TM>
    void popAndGoLeft(node<TM> *&n) {
      node<TM> *left = n->left;

      delete n->pop();

      n = left;
    }

    template <class TM>
    node<TM>* firstNode(node<TM> *n) {
      if ((n == NULL) ||
         (n->left == NULL))
        return n;

      node<TM> *end = n;

      while(end->left)
        end = end->left;

      return end;
    }

    template <class TM>
    node<TM>* lastNode(node<TM> *n) {
      if ((n == NULL) ||
         (n->right == NULL))
        return n;

      node<TM> *end = n;

      while(end->right)
        end = end->right;

      return end;
    }

    template <class TM>
    int length(node<TM> *n) {
      int l = 0;

      while(n) {
        ++l;
        n = n->right;
      }

      return l;
    }
    //==============================================
  }
}
