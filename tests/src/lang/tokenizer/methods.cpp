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

#include <occa/tools/testing.hpp>

#include <occa/lang/token.hpp>
#include <occa/lang/tokenizer.hpp>
#include <occa/lang/processingStages.hpp>

void testSkipMethods();
void testPushPop();
void testPeekMethods();
void testTokenMethods();
void testCommentSkipping();
void testStringMethods();
void testStringMerging();
void testExternMerging();
void testPrimitiveMethods();
void testErrors();

using namespace occa::lang;


//---[ Tests ]--------------------------
int main(const int argc, const char **argv) {
  testSkipMethods();
  testPushPop();
  testPeekMethods();
  testTokenMethods();
  testCommentSkipping();
  testStringMethods();
  testStringMerging();
  testExternMerging();
  testPrimitiveMethods();
  testErrors();

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

void testPeekMethods() {
  setStream("abcd");
  ASSERT_EQ_BINARY(
    tokenType::identifier,
    tokenizer.peek()
  );
  setStream("_abcd");
  ASSERT_EQ_BINARY(
    tokenType::identifier,
    tokenizer.peek()
  );
  setStream("_abcd020230");
  ASSERT_EQ_BINARY(
    tokenType::identifier,
    tokenizer.peek()
  );

  setStream("1");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    tokenizer.peek()
  );
  setStream("2.0");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    tokenizer.peek()
  );
  setStream("2.0e10");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    tokenizer.peek()
  );
  setStream(".0e10-10");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    tokenizer.peek()
  );

  setStream("+");
  ASSERT_EQ_BINARY(
    tokenType::op,
    tokenizer.peek()
  );
  setStream("<");
  ASSERT_EQ_BINARY(
    tokenType::op,
    tokenizer.peek()
  );
  setStream("->*");
  ASSERT_EQ_BINARY(
    tokenType::op,
    tokenizer.peek()
  );
  setStream("==");
  ASSERT_EQ_BINARY(
    tokenType::op,
    tokenizer.peek()
  );

  testStringPeek("\"\""                , 0);
  testStringPeek("\"string\\\"string\"", 0);
  testStringPeek("R\"(string)\""       , encodingType::R);
  testStringPeek("u8\"string\""        , encodingType::u8);
  testStringPeek("u\"string\""         , encodingType::u);
  testStringPeek("U\"string\""         , encodingType::U);
  testStringPeek("L\"string\""         , encodingType::L);
  testStringPeek("u8R\"(string)\""     , (encodingType::u8 |
                                          encodingType::R));
  testStringPeek("uR\"(string)\""      , (encodingType::u |
                                          encodingType::R));
  testStringPeek("UR\"(string)\""      , (encodingType::U |
                                          encodingType::R));
  testStringPeek("LR\"(string)\""      , (encodingType::L |
                                          encodingType::R));

  testCharPeek("'a'"    , 0);
  testCharPeek("'\\''"  , 0);
  testCharPeek("u'\\''" , encodingType::u);
  testCharPeek("U'\\''" , encodingType::U);
  testCharPeek("L'\\''" , encodingType::L);

  ASSERT_EQ("foobar",
            getHeader("<foobar>"));
  ASSERT_EQ("foobar",
            getHeader("\"foobar\""));
}
//======================================
