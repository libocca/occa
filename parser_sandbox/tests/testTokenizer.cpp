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

void testCharPeek(const std::string &s) {
  setStream(s);
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::withUType |
                    occa::lang::tokenType::char_,
                    stream.peek());
}

void testStringPeek(const std::string &s) {
  setStream(s);
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::withUType |
                    occa::lang::tokenType::string,
                    stream.peek());
}

void testPeekMethods() {
  setStream("abcd");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::identifier,
                    stream.peek());
  setStream("_abcd");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::identifier,
                    stream.peek());
  setStream("_abcd020230");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::identifier,
                    stream.peek());

  setStream("<foobar>");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::systemHeader,
                    stream.peekForHeader());
  setStream("\"foobar\"");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::header,
                    stream.peekForHeader());

  setStream("1");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::primitive,
                    stream.peek());
  setStream("+2.0");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::primitive,
                    stream.peek());
  setStream("+2.0e10");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::primitive,
                    stream.peek());
  setStream("+.0e10-10");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::primitive,
                    stream.peek());

  setStream("+");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::op,
                    stream.peek());
  setStream("<");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::op,
                    stream.peek());
  setStream("->*");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::op,
                    stream.peek());
  setStream("==");
  OCCA_ASSERT_EQUAL(occa::lang::tokenType::op,
                    stream.peek());

  testCharPeek("'a'");
  testCharPeek("'\\''");
  testCharPeek("u8'\\''");
  testCharPeek("u'\\''");
  testCharPeek("U'\\''");
  testCharPeek("L'\\''");

  testStringPeek("");
  testStringPeek("\"string\\\"string\"");
  testStringPeek("R\"string\"");
  testStringPeek("u8\"string\"");
  testStringPeek("u\"string\"");
  testStringPeek("U\"string\"");
  testStringPeek("L\"string\"");
  testStringPeek("u8R\"string\"");
  testStringPeek("uR\"string\"");
  testStringPeek("UR\"string\"");
  testStringPeek("LR\"string\"");

  //---[ Failures ]---------------------
  // setStream("abcd");
  // OCCA_ASSERT_EQUAL(occa::lang::tokenType::identifier,
  //                   stream.peek());
  // setStream("_abcd");
  // OCCA_ASSERT_EQUAL(occa::lang::tokenType::identifier,
  //                   stream.peek());
  // setStream("_abcd020230");
  // OCCA_ASSERT_EQUAL(occa::lang::tokenType::identifier,
  //                   stream.peek());
  // occa::lang::tokenType::none;
}

void testTokenMethods() {
}
