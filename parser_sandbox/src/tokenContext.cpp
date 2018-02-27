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
#include "token.hpp"
#include "tokenContext.hpp"

namespace occa {
  namespace lang {
    tokenPosition::tokenPosition() :
      start(0),
      pos(0),
      end(0) {}

    tokenPosition::tokenPosition(const int start_,
                                 const int end_) :
      start(start_),
      pos(start_),
      end(end_) {}

    tokenPosition::tokenPosition(const int start_,
                                 const int pos_,
                                 const int end_) :
      start(start_),
      pos(pos_),
      end(end_) {}

    tokenContext::tokenContext() {}

    tokenContext::tokenContext(const tokenVector &tokens_) :
      tokens(tokens_),
      tp(0, (int) tokens.size()) {}

    tokenContext::~tokenContext() {
      clear();
    }

    void tokenContext::clear() {
      tp.start = 0;
      tp.pos   = 0;
      tp.end   = 0;

      const int tokenCount = (int) tokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        delete tokens[i];
      }
      tokens.clear();
      stack.clear();
    }

    void tokenContext::set(const tokenVector &tokens_) {
      clear();

      tokens = tokens_;

      tp.start = 0;
      tp.pos   = 0;
      tp.end   = (int) tokens.size();
    }

    void tokenContext::push(const int start,
                            const int end) {
      stack.push_back(tp);
      tp.start = start;
      tp.pos   = start;
      tp.end   = end;
    }

    tokenPosition tokenContext::pop() {
      OCCA_ERROR("Unable to call tokenContext::pop",
                 stack.size());
      tp = stack.back();
      stack.pop_back();
      return tp;
    }
  }
}
