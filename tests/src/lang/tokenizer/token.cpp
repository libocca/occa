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

void testPeekMethods();
void testTokenMethods();

using namespace occa::lang;

int main(const int argc, const char **argv) {
  testPeekMethods();
  testTokenMethods();

  return 0;
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

void testTokenMethods() {
  setToken("abcd");
  ASSERT_EQ_BINARY(
    tokenType::identifier,
    getTokenType()
  );
  setToken("_abcd");
  ASSERT_EQ_BINARY(
    tokenType::identifier,
    getTokenType()
  );
  setToken("_abcd020230");
  ASSERT_EQ_BINARY(
    tokenType::identifier,
    getTokenType()
  );

  setToken("true");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    getTokenType()
  );
  setToken("false");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    getTokenType()
  );

  setToken("1");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    getTokenType()
  );
  setToken("2.0");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    getTokenType()
  );
  setToken("2.0e10");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    getTokenType()
  );
  setToken(".0e10-10");
  ASSERT_EQ_BINARY(
    tokenType::primitive,
    getTokenType()
  );

  setToken("+");
  ASSERT_EQ_BINARY(
    tokenType::op,
    getTokenType()
  );
  setToken("<");
  ASSERT_EQ_BINARY(
    tokenType::op,
    getTokenType()
  );
  setToken("->*");
  ASSERT_EQ_BINARY(
    tokenType::op,
    getTokenType()
  );
  setToken("==");
  ASSERT_EQ_BINARY(
    tokenType::op,
    getTokenType()
  );

  testStringToken("\"\""                , 0);
  testStringToken("\"string\\\"string\"", 0);
  testStringToken("R\"(string)\""       , encodingType::R);
  testStringToken("u8\"string\""        , encodingType::u8);
  testStringToken("u\"string\""         , encodingType::u);
  testStringToken("U\"string\""         , encodingType::U);
  testStringToken("L\"string\""         , encodingType::L);
  testStringToken("u8R\"(string)\""     , (encodingType::u8 |
                                           encodingType::R));
  testStringToken("uR\"(string)\""      , (encodingType::u |
                                           encodingType::R));
  testStringToken("UR\"(string)\""      , (encodingType::U |
                                           encodingType::R));
  testStringToken("LR\"(string)\""      , (encodingType::L |
                                           encodingType::R));

  testCharToken("'a'"    , 0);
  testCharToken("'\\''"  , 0);
  testCharToken("u'\\''" , encodingType::u);
  testCharToken("U'\\''" , encodingType::U);
  testCharToken("L'\\''" , encodingType::L);
}
