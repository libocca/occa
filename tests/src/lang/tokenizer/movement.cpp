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
#include "utils.hpp"

void testSkipMethods();
void testPushPop();

using namespace occa::lang;

int main(const int argc, const char **argv) {
  testSkipMethods();
  testPushPop();

  return 0;
}

void testSkipMethods() {
  setStream("ab\nc\n\n\n\n\n\nd\ne");
  const char *c = source.c_str();

  tokenizer.skipTo('a');
  ASSERT_EQ('a', *tokenizer.fp.start);

  tokenizer.skipTo('b');
  ASSERT_EQ('b' , *tokenizer.fp.start);

  tokenizer.skipTo('e');
  ASSERT_EQ('e' , *tokenizer.fp.start);

  tokenizer.fp.start = c;
  tokenizer.skipTo("c\n");
  ASSERT_EQ(c + 2, tokenizer.fp.start);

  tokenizer.fp.start = c + 6;
  tokenizer.skipFrom("\n");
  ASSERT_EQ('d' , *tokenizer.fp.start);
}

void testPushPop() {
  setStream("a\nb\nc\nd\ne");
  const char *c = source.c_str();

  tokenizer.push();
  tokenizer.skipTo('c');
  ASSERT_EQ(3,
            tokenizer.fp.line);
  ASSERT_EQ(c + 4,
            tokenizer.fp.start);
  tokenizer.popAndRewind();
  ASSERT_EQ(1,
            tokenizer.fp.line);
  ASSERT_EQ(c + 0,
            tokenizer.fp.start);
  tokenizer.push();
  tokenizer.push();
  tokenizer.push();
  tokenizer.skipTo('c');
  tokenizer.pop();
  tokenizer.pop();
  tokenizer.pop();
  ASSERT_EQ(3,
            tokenizer.fp.line);
  ASSERT_EQ(c + 4,
            tokenizer.fp.start);
}
