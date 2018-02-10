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

#include "token.hpp"
#include "tokenizer.hpp"
#include "stringTokenMerger.hpp"

void testSkipMethods();
void testPushPop();
void testPeekMethods();
void testTokenMethods();
void testCommentSkipping();
void testStringMethods();
void testStringMerging();
void testPrimitiveMethods();
void testErrors();

std::string source;
occa::lang::tokenizer tokenizer(NULL);
occa::stream<occa::lang::token_t*> mergeTokenStream;
occa::lang::token_t *token = NULL;

void setStream(const std::string &s) {
  source = s;
  tokenizer = occa::lang::tokenizer(source.c_str());
  mergeTokenStream = tokenizer.map(new occa::lang::stringTokenMerger());
}

void getToken() {
  delete token;
  tokenizer >> token;
}

void getStringMergeToken() {
  delete token;
  mergeTokenStream >> token;
}

void setToken(const std::string &s) {
  setStream(s);
  getToken();
}

void setHeaderToken(const std::string &s) {
  setStream(s);
  delete token;
  token = tokenizer.getHeaderToken();
}

int tokenType() {
  return token ? token->type() : 0;
}

int main(const int argc, const char **argv) {
  testSkipMethods();
  testPushPop();
  testPeekMethods();
  testTokenMethods();
  testCommentSkipping();
  testStringMethods();
  testPrimitiveMethods();
  testErrors();

  delete token;

  return 0;
}

void testSkipMethods() {
  setStream("ab\nc\n\n\n\n\n\nd\ne");
  const char *c = source.c_str();

  tokenizer.skipTo('a');
  OCCA_ASSERT_EQUAL('a', *tokenizer.fp.pos);

  tokenizer.skipTo('b');
  OCCA_ASSERT_EQUAL('b', *tokenizer.fp.pos);

  tokenizer.skipTo('e');
  OCCA_ASSERT_EQUAL('e', *tokenizer.fp.pos);

  tokenizer.fp.pos = c;
  tokenizer.skipTo("c\n");
  OCCA_ASSERT_EQUAL(c + 2, tokenizer.fp.pos);

  tokenizer.fp.pos = c + 6;
  tokenizer.skipFrom("\n");
  OCCA_ASSERT_EQUAL('d', *tokenizer.fp.pos);
}

void testPushPop() {
  setStream("a\nb\nc\nd\ne");
  const char *c = source.c_str();

  tokenizer.push();
  tokenizer.skipTo('c');
  OCCA_ASSERT_EQUAL(3,
                    tokenizer.fp.line);
  OCCA_ASSERT_EQUAL(c + 4,
                    tokenizer.fp.pos);
  tokenizer.popAndRewind();
  OCCA_ASSERT_EQUAL(1,
                    tokenizer.fp.line);
  OCCA_ASSERT_EQUAL(c + 0,
                    tokenizer.fp.pos);
  tokenizer.push();
  tokenizer.push();
  tokenizer.push();
  tokenizer.skipTo('c');
  tokenizer.pop();
  tokenizer.pop();
  tokenizer.pop();
  OCCA_ASSERT_EQUAL(3,
                    tokenizer.fp.line);
  OCCA_ASSERT_EQUAL(c + 4,
                    tokenizer.fp.pos);
}

#define testStringPeek(s, encoding_)                      \
  setStream(s);                                           \
  OCCA_ASSERT_EQUAL_BINARY(                               \
    (encoding_ << occa::lang::tokenType::encodingShift) | \
    occa::lang::tokenType::string,                        \
    tokenizer.peek())

#define testCharPeek(s, encoding_)                        \
  setStream(s);                                           \
  OCCA_ASSERT_EQUAL_BINARY(                               \
    (encoding_ << occa::lang::tokenType::encodingShift) | \
    occa::lang::tokenType::char_,                         \
    tokenizer.peek())

void testPeekMethods() {
  setStream("abcd");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tokenizer.peek()
  );
  setStream("_abcd");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tokenizer.peek()
  );
  setStream("_abcd020230");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tokenizer.peek()
  );

  setStream("1");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tokenizer.peek()
  );
  setStream("2.0");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tokenizer.peek()
  );
  setStream("2.0e10");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tokenizer.peek()
  );
  setStream(".0e10-10");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tokenizer.peek()
  );

  setStream("+");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tokenizer.peek()
  );
  setStream("<");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tokenizer.peek()
  );
  setStream("->*");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tokenizer.peek()
  );
  setStream("==");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tokenizer.peek()
  );

  testStringPeek("\"\""                , 0);
  testStringPeek("\"string\\\"string\"", 0);
  testStringPeek("R\"(string)\""       , occa::lang::encodingType::R);
  testStringPeek("u8\"string\""        , occa::lang::encodingType::u8);
  testStringPeek("u\"string\""         , occa::lang::encodingType::u);
  testStringPeek("U\"string\""         , occa::lang::encodingType::U);
  testStringPeek("L\"string\""         , occa::lang::encodingType::L);
  testStringPeek("u8R\"(string)\""     , (occa::lang::encodingType::u8 |
                                          occa::lang::encodingType::R));
  testStringPeek("uR\"(string)\""      , (occa::lang::encodingType::u |
                                          occa::lang::encodingType::R));
  testStringPeek("UR\"(string)\""      , (occa::lang::encodingType::U |
                                          occa::lang::encodingType::R));
  testStringPeek("LR\"(string)\""      , (occa::lang::encodingType::L |
                                          occa::lang::encodingType::R));

  testCharPeek("'a'"    , 0);
  testCharPeek("'\\''"  , 0);
  testCharPeek("u'\\''" , occa::lang::encodingType::u);
  testCharPeek("U'\\''" , occa::lang::encodingType::U);
  testCharPeek("L'\\''" , occa::lang::encodingType::L);

  setStream("<foobar>");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::systemHeader,
                           tokenizer.peekForHeader());
  setStream("\"foobar\"");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::header,
                           tokenizer.peekForHeader());
}

#define testStringToken(s, encoding_)                                   \
  setToken(s);                                                          \
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::string,               \
                           tokenType())                                 \
  OCCA_ASSERT_EQUAL_BINARY(encoding_,                                   \
                           token->to<occa::lang::stringToken>().encoding)

#define testCharToken(s, encoding_)                                     \
  setToken(s);                                                          \
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::char_,                \
                           tokenType())                                 \
  OCCA_ASSERT_EQUAL_BINARY(encoding_,                                   \
                           token->to<occa::lang::charToken>().encoding)

void testTokenMethods() {
  setToken("abcd");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tokenType()
  );
  setToken("_abcd");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tokenType()
  );
  setToken("_abcd020230");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tokenType()
  );

  setToken("1");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tokenType()
  );
  setToken("2.0");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tokenType()
  );
  setToken("2.0e10");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tokenType()
  );
  setToken(".0e10-10");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tokenType()
  );

  setToken("+");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tokenType()
  );
  setToken("<");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tokenType()
  );
  setToken("->*");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tokenType()
  );
  setToken("==");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tokenType()
  );

  testStringToken("\"\""                , 0);
  testStringToken("\"string\\\"string\"", 0);
  testStringToken("R\"(string)\""       , occa::lang::encodingType::R);
  testStringToken("u8\"string\""        , occa::lang::encodingType::u8);
  testStringToken("u\"string\""         , occa::lang::encodingType::u);
  testStringToken("U\"string\""         , occa::lang::encodingType::U);
  testStringToken("L\"string\""         , occa::lang::encodingType::L);
  testStringToken("u8R\"(string)\""     , (occa::lang::encodingType::u8 |
                                           occa::lang::encodingType::R));
  testStringToken("uR\"(string)\""      , (occa::lang::encodingType::u |
                                           occa::lang::encodingType::R));
  testStringToken("UR\"(string)\""      , (occa::lang::encodingType::U |
                                           occa::lang::encodingType::R));
  testStringToken("LR\"(string)\""      , (occa::lang::encodingType::L |
                                           occa::lang::encodingType::R));

  testCharToken("'a'"    , 0);
  testCharToken("'\\''"  , 0);
  testCharToken("u'\\''" , occa::lang::encodingType::u);
  testCharToken("U'\\''" , occa::lang::encodingType::U);
  testCharToken("L'\\''" , occa::lang::encodingType::L);

  // Test this in testPreprocessor (#include tests)
  setHeaderToken("<foobar>");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::header,
    tokenType()
  );
  OCCA_ASSERT_TRUE(token->to<occa::lang::headerToken>().systemHeader);
  setHeaderToken("\"foobar\"");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::header,
    tokenType()
  );
  OCCA_ASSERT_FALSE(token->to<occa::lang::headerToken>().systemHeader);
}

#define testStringValue(s, value_)              \
  setStream(s);                                 \
  getToken();                                   \
  testNextStringValue(value_)

#define testStringMergeValue(s, value_)         \
  setStream(s);                                 \
  getStringMergeToken();                        \
  testNextStringValue(value_)

#define testNextStringValue(value_)             \
  OCCA_ASSERT_EQUAL_BINARY(                     \
    occa::lang::tokenType::string,              \
    tokenType()                                 \
  );                                            \
  OCCA_ASSERT_EQUAL(                            \
    value_,                                     \
    token->to<occa::lang::stringToken>().value  \
  )


void addEncodingPrefixes(std::string &s1,
                         std::string &s2,
                         std::string &s3,
                         const std::string &encoding1,
                         const std::string &encoding2) {
  const std::string a  = "\"a\"";
  const std::string b  = "\"b\"";
  s1 = (encoding1 + a) + " " + b;
  s2 = a               + " " + (encoding2 + b);
  s3 = (encoding1 + a) + " " + (encoding2 + b);
}

void testCommentSkipping() {
  setStream("// test    \n"
            "1.0");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::newline,
    tokenizer.peek()
  );
  getToken();
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tokenizer.peek()
  );
  setStream("/* 2.0 */ foo");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tokenizer.peek()
  );
  setStream("/*         \n"
            "2.0        \n"
            "*/ foo");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tokenizer.peek()
  );
}

void testStringMethods() {
  // Test values
  testStringValue("\"\""                , "");
  testStringValue("\"string\\\"string\"", "string\\\"string");
  testStringValue("R\"(string)\""       , "string");
  testStringValue("u8\"string\""        , "string");
  testStringValue("u\"string\""         , "string");
  testStringValue("U\"string\""         , "string");
  testStringValue("L\"string\""         , "string");
  testStringValue("u8R\"(string)\""     , "string");
  testStringValue("uR\"(string)\""      , "string");
  testStringValue("UR\"(string)\""      , "string");
  testStringValue("LR\"(string)\""      , "string");

  // Test raw strings
  testStringValue("R\"*(string)*\""  , "string");
  testStringValue("u8R\"*(string)*\"", "string");
  testStringValue("uR\"*(string)*\"" , "string");
  testStringValue("UR\"*(string)*\"" , "string");
  testStringValue("LR\"*(string)*\"" , "string");

  testStringValue("R\"foo(string)foo\""  , "string");
  testStringValue("u8R\"foo(string)foo\"", "string");
  testStringValue("uR\"foo(string)foo\"" , "string");
  testStringValue("UR\"foo(string)foo\"" , "string");
  testStringValue("LR\"foo(string)foo\"" , "string");
}

void testStringMerging() {
  // Test string concatination
  const std::string ab = "ab";
  std::string s1, s2, s3;
  addEncodingPrefixes(s1, s2, s3, "", "");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringToken(s1, 0);
  testStringToken(s2, 0);
  testStringToken(s3, 0);

  addEncodingPrefixes(s1, s2, s3, "", "u");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringToken(s1, 0);
  testStringToken(s2, occa::lang::encodingType::u);
  testStringToken(s3, occa::lang::encodingType::u);

  addEncodingPrefixes(s1, s2, s3, "", "U");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringToken(s1, 0);
  testStringToken(s2, occa::lang::encodingType::U);
  testStringToken(s3, occa::lang::encodingType::U);

  addEncodingPrefixes(s1, s2, s3, "", "L");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringToken(s1, 0);
  testStringToken(s2, occa::lang::encodingType::L);
  testStringToken(s3, occa::lang::encodingType::L);

  addEncodingPrefixes(s1, s2, s3, "u", "U");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringToken(s1, occa::lang::encodingType::u);
  testStringToken(s2, occa::lang::encodingType::U);
  testStringToken(s3, occa::lang::encodingType::U);

  addEncodingPrefixes(s1, s2, s3, "u", "L");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringToken(s1, occa::lang::encodingType::u);
  testStringToken(s2, occa::lang::encodingType::L);
  testStringToken(s3, occa::lang::encodingType::L);

  addEncodingPrefixes(s1, s2, s3, "U", "L");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringToken(s1, occa::lang::encodingType::U);
  testStringToken(s2, occa::lang::encodingType::L);
  testStringToken(s3, occa::lang::encodingType::L);
}

#define testPrimitiveToken(type_, value_)                 \
  getToken();                                             \
  OCCA_ASSERT_EQUAL_BINARY(                               \
    occa::lang::tokenType::primitive,                     \
    tokenType());                                         \
  OCCA_ASSERT_EQUAL(                                      \
    value_,                                               \
    (type_) token->to<occa::lang::primitiveToken>().value \
  )

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
  setStream("$ test\n\"foo\" \"bar\" ` bar foo");
  std::cerr << "Testing error outputs:\n";
  getToken();
  tokenizer.skipTo('\n');
  getToken();
  getToken();
  getToken();
}
