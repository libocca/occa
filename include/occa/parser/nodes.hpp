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

#ifndef OCCA_PARSER_NODES_HEADER
#define OCCA_PARSER_NODES_HEADER

#include "occa/defines.hpp"
#include "occa/parser/defines.hpp"

namespace occa {
  namespace parserNS {
    //---[ Node ]-----------------------------------
    template <class TM>
    class node {
    public:
      node *left, *right, *up, *down;
      TM value;

      node();
      node(const TM &t);
      node(const node<TM> &n);

      node& operator = (const node<TM> &n);

      node<TM>* pop();

      node* push(node<TM> *n);
      node* push(node<TM> *nStart, node<TM> *nEnd);
      node* push(const TM &t);

      node* pushDown(node<TM> *n);
      node* pushDown(const TM &t);

      void print(const std::string &tab = "");

      void printPtr(const std::string &tab = "");
    };

    template <class TM>
    void popAndGoRight(node<TM> *&n);

    template <class TM>
    void popAndGoLeft(node<TM> *&n);

    template <class TM>
    node<TM>* firstNode(node<TM> *n);

    template <class TM>
    node<TM>* lastNode(node<TM> *n);

    template <class TM>
    int length(node<TM> *n);
    //==============================================
  }
}

#include "occa/parser/nodes.tpp"

#endif
