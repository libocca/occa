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
#include "occa/tools/testing.hpp"

#include "tokenizer.hpp"

void testSkipMethods();
void testPushPop();
void testPeekMethods();
void testTokenMethods();

std::string streamSource;
occa::lang::charStream stream(NULL);

int main(const int argc, const char **argv) {
  testSkipMethods();
  testPushPop();
  testPeekMethods();
  testTokenMethods();
  return 0;
}

void setStream(const std::string &s) {
  streamSource = s;
  stream = occa::lang::charStream(NULL, streamSource.c_str());
}

void testSkipMethods() {
  setStream("a\nb\nc\n\n\n\n\n\nd\ne");
  const char *c = streamSource.c_str();

  stream.skipTo('b');
  OCCA_ASSERT_EQUAL('b', *stream.getPosition());
  stream.skipTo('e');
  OCCA_ASSERT_EQUAL('e', *stream.getPosition());

  stream.setPosition(c);
  stream.skipTo("c\n");
  OCCA_ASSERT_EQUAL(c + 1, stream.getPosition());

  stream.setPosition(c + 5);
  stream.skipFrom("\n");
  OCCA_ASSERT_EQUAL('d', *stream.getPosition());
}

void testPushPop() {
  setStream("a\nb\nc\nd\ne");
  const char *c = streamSource.c_str();

  stream.push();
  stream.skipTo('c');
  OCCA_ASSERT_EQUAL(3,
                    stream.getLine());
  OCCA_ASSERT_EQUAL(c + 4,
                    stream.getPosition());
  stream.popAndRewind();
  OCCA_ASSERT_EQUAL(1,
                    stream.getLine());
  OCCA_ASSERT_EQUAL(c + 0,
                    stream.getPosition());
  stream.push();
  stream.push();
  stream.push();
  stream.skipTo('c');
  stream.pop();
  stream.pop();
  stream.pop();
  OCCA_ASSERT_EQUAL(3,
                    stream.getLine());
  OCCA_ASSERT_EQUAL(c + 4,
                    stream.getPosition());
}

#define testCharPeek(s, type)                                 \
  setStream(s);                                               \
  OCCA_ASSERT_EQUAL_BINARY(type |                             \
                           occa::lang::tokenType::char_,      \
                           stream.peek())

#define testStringPeek(s, type)                               \
  setStream(s);                                               \
  OCCA_ASSERT_EQUAL_BINARY(type |                             \
                           occa::lang::tokenType::string,     \
                           stream.peek())

void testPeekMethods() {
  setStream("abcd");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
                           stream.peek());
  setStream("_abcd");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
                           stream.peek());
  setStream("_abcd020230");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
                           stream.peek());

  setStream("<foobar>");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::systemHeader,
                           stream.peekForHeader());
  setStream("\"foobar\"");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::header,
                           stream.peekForHeader());

  setStream("1");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           stream.peek());
  setStream("+2.0");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           stream.peek());
  setStream("+2.0e10");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           stream.peek());
  setStream("+.0e10-10");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           stream.peek());

  setStream("+");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           stream.peek());
  setStream("<");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           stream.peek());
  setStream("->*");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           stream.peek());
  setStream("==");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           stream.peek());

  testCharPeek("'a'"    , 0);
  testCharPeek("'\\''"  , 0);
  testCharPeek("u'\\''" , occa::lang::tokenType::withUType);
  testCharPeek("U'\\''" , occa::lang::tokenType::withUType);
  testCharPeek("L'\\''" , occa::lang::tokenType::withUType);

  testStringPeek("\"\""                , 0);
  testStringPeek("\"string\\\"string\"", 0);
  testStringPeek("R\"string\""         , occa::lang::tokenType::withUType);
  testStringPeek("u8\"string\""        , occa::lang::tokenType::withUType);
  testStringPeek("u\"string\""         , occa::lang::tokenType::withUType);
  testStringPeek("U\"string\""         , occa::lang::tokenType::withUType);
  testStringPeek("L\"string\""         , occa::lang::tokenType::withUType);
  testStringPeek("u8R\"string\""       , occa::lang::tokenType::withUType);
  testStringPeek("uR\"string\""        , occa::lang::tokenType::withUType);
  testStringPeek("UR\"string\""        , occa::lang::tokenType::withUType);
  testStringPeek("LR\"string\""        , occa::lang::tokenType::withUType);

  //---[ Failures ]---------------------
  // setStream("abcd");
  // OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
  //                   stream.peek());
  // setStream("_abcd");
  // OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
  //                   stream.peek());
  // setStream("_abcd020230");
  // OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
  //                   stream.peek());
  // occa::lang::tokenType::none;
}

void testTokenMethods() {
}
