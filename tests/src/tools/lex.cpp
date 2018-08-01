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
#include <occa.hpp>
#include <occa/tools/testing.hpp>

void testCharsets();
void testSkipTo();
void testWhitespaceMethods();

int main(const int argc, const char **argv) {
  testCharsets();
  testSkipTo();
  testWhitespaceMethods();

  return 0;
}

void testCharsets() {
  for (int i = 0; i < 10; ++i) {
    char c = '0' + i;
    ASSERT_TRUE(occa::lex::isDigit(c));
    ASSERT_FALSE(occa::lex::isAlpha(c));
    ASSERT_TRUE(occa::lex::inCharset(c, occa::lex::numberCharset));
  }
  for (int i = 0; i < 26; ++i) {
    char c = 'a' + i;
    ASSERT_FALSE(occa::lex::isDigit(c));
    ASSERT_TRUE(occa::lex::isAlpha(c));
    ASSERT_FALSE(occa::lex::inCharset(c, occa::lex::numberCharset));

    c = 'A' + i;
    ASSERT_FALSE(occa::lex::isDigit(c));
    ASSERT_TRUE(occa::lex::isAlpha(c));
    ASSERT_FALSE(occa::lex::inCharset(c, occa::lex::numberCharset));
  }

  ASSERT_TRUE(occa::lex::inCharset('a', "abc"));
  ASSERT_FALSE(occa::lex::inCharset('d', "abc"));
}

void testSkipTo() {
}

void testWhitespaceMethods() {
}
