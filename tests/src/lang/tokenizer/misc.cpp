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

void testCommentSkipping();
void testExternMerging();
void testPrimitiveMethods();
void testErrors();

int main(const int argc, const char **argv) {
  testCommentSkipping();
  testExternMerging();
  testPrimitiveMethods();
  testErrors();

  return 0;
}

void testCommentSkipping() {
  // [1.0]
  setStream("// test    \n"
            "1.0");
  ASSERT_EQ_BINARY(
    tokenType::newline,
    tokenizer.peek()
  );
  getToken();
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    tokenizer.peek()
  );

  // [ foo]
  setStream("/* 2.0 */ foo");
  ASSERT_EQ_BINARY(
    tokenType::identifier,
    tokenizer.peek()
  );

  // [ foo]
  setStream("/*         \n"
            "2.0        \n"
            "*/ foo");
  ASSERT_EQ_BINARY(
    tokenType::identifier,
    tokenizer.peek()
  );

  // [*/b]
  setStream("/*a*/*/b/*c*/");
  ASSERT_EQ_BINARY(
    tokenType::op,
    tokenizer.peek()
  );
  getToken();
  ASSERT_EQ_BINARY(
    tokenType::op,
    tokenizer.peek()
  );
  getToken();
  ASSERT_EQ_BINARY(
    tokenType::identifier,
    tokenizer.peek()
  );
}

void testExternMerging() {
  setStream("extern"
            " extern \"C\""
            " extern \"C++\""
            " extern \"Cpp\"");
  getMergedToken();
  ASSERT_EQ_BINARY(tokenType::identifier,
                   getTokenType());
  ASSERT_EQ("extern",
            token->to<identifierToken>().value);

  getMergedToken();
  ASSERT_EQ_BINARY(tokenType::identifier,
                   getTokenType());
  ASSERT_EQ("extern \"C\"",
            token->to<identifierToken>().value);

  getMergedToken();
  ASSERT_EQ_BINARY(tokenType::identifier,
                   getTokenType());
  ASSERT_EQ("extern \"C++\"",
            token->to<identifierToken>().value);

  getMergedToken();
  ASSERT_EQ_BINARY(tokenType::identifier,
                   getTokenType());
  ASSERT_EQ("extern",
            token->to<identifierToken>().value);

  getMergedToken();
  testNextStringValue("Cpp");
}

void testPrimitiveMethods() {
  setStream("1 68719476735L +0.1 .1e-10 -4.5L");
  testPrimitiveToken(int, 1);
  testPrimitiveToken(int64_t, 68719476735L);
  getToken();
  testPrimitiveToken(float, 0.1);
  testPrimitiveToken(float, .1e-10);
  getToken();
  testPrimitiveToken(double, 4.5L);
}

void testErrors() {
  unknownTokenFilter unknownFilter(true);

  setStream("$ test\n\"foo\" \"bar\" ` bar foo");
  occa::lang::stream<token_t*> streamWithErrors = (
    tokenizer.filter(unknownFilter)
  );

  std::cerr << "Testing error outputs:\n";
  while (!streamWithErrors.isEmpty()) {
    token = NULL;
    streamWithErrors >> token;
    delete token;
  }
}
