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
#include <sstream>

#include "occa/tools/testing.hpp"

#include "token.hpp"
#include "tokenContext.hpp"

using namespace occa::lang;

void testMethods();

int main(const int argc, const char **argv) {
  testMethods();

  return 0;
}

void testMethods() {
  tokenContext ctx;
  OCCA_ASSERT_EQUAL(0, ctx.tp.start);
  OCCA_ASSERT_EQUAL(0, ctx.tp.pos);
  OCCA_ASSERT_EQUAL(0, ctx.tp.end);

  tokenVector vec;
  vec.push_back(NULL);
  vec.push_back(NULL);
  vec.push_back(NULL);

  ctx.set(vec);
  OCCA_ASSERT_EQUAL(0, ctx.tp.start);
  OCCA_ASSERT_EQUAL(0, ctx.tp.pos);
  OCCA_ASSERT_EQUAL(3, ctx.tp.end);

  ctx.push(1, 2);
  OCCA_ASSERT_EQUAL(1, ctx.tp.start);
  OCCA_ASSERT_EQUAL(1, ctx.tp.pos);
  OCCA_ASSERT_EQUAL(2, ctx.tp.end);

  ctx.pop();
  OCCA_ASSERT_EQUAL(0, ctx.tp.start);
  OCCA_ASSERT_EQUAL(0, ctx.tp.pos);
  OCCA_ASSERT_EQUAL(3, ctx.tp.end);
}
