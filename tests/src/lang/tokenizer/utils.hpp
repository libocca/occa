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
#ifndef OCCA_TESTS_LANG_TOKENIZER_UTILS
#define OCCA_TESTS_LANG_TOKENIZER_UTILS

#include <occa/tools/testing.hpp>

#include <occa/lang/token.hpp>
#include <occa/lang/tokenizer.hpp>
#include <occa/lang/processingStages.hpp>

using namespace occa::lang;

//---[ Util Methods ]-------------------
std::string source;
tokenizer_t tokenizer;
stringTokenMerger stringMerger;
externTokenMerger externMerger;
occa::lang::stream<token_t*> mergeTokenStream = (
  tokenizer
  .map(stringMerger)
  .map(externMerger)
);
token_t *token = NULL;

void setStream(const std::string &s) {
  delete token;
  token = NULL;
  source = s;
  tokenizer.set(source.c_str());
}

void getToken() {
  delete token;
  token = NULL;
  tokenizer >> token;
}

void getMergedToken() {
  delete token;
  token = NULL;
  mergeTokenStream >> token;
}

void setToken(const std::string &s) {
  setStream(s);
  getToken();
}

std::string getHeader(const std::string &s) {
  setStream(s);
  return tokenizer.getHeader();
}

int getTokenType() {
  return token ? token->type() : 0;
}
//======================================

//---[ Macro Util Methods ]-------------
#define testStringPeek(s, encoding_)            \
  setStream(s);                                 \
  ASSERT_EQ_BINARY(                             \
    (encoding_ << tokenType::encodingShift) |   \
    tokenType::string,                          \
    tokenizer.peek())

#define testCharPeek(s, encoding_)              \
  setStream(s);                                 \
  ASSERT_EQ_BINARY(                             \
    (encoding_ << tokenType::encodingShift) |   \
    tokenType::char_,                           \
    tokenizer.peek())

#define testStringToken(s, encoding_)                 \
  setToken(s);                                        \
  ASSERT_EQ_BINARY(tokenType::string,                 \
                   getTokenType());                   \
  ASSERT_EQ_BINARY(encoding_,                         \
                   token->to<stringToken>().encoding)

#define testStringMergeToken(s, encoding_)            \
  setStream(s);                                       \
  getMergedToken();                                   \
  ASSERT_EQ_BINARY(tokenType::string,                 \
                   getTokenType());                   \
  ASSERT_EQ_BINARY(encoding_,                         \
                   token->to<stringToken>().encoding)

#define testCharToken(s, encoding_)                 \
  setToken(s);                                      \
  ASSERT_EQ_BINARY(tokenType::char_,                \
                   getTokenType())                  \
  ASSERT_EQ_BINARY(encoding_,                       \
                   token->to<charToken>().encoding)

#define testStringValue(s, value_)              \
  setStream(s);                                 \
  getToken();                                   \
  testNextStringValue(value_)

#define testStringMergeValue(s, value_)         \
  setStream(s);                                 \
  getMergedToken();                             \
  testNextStringValue(value_)

#define testNextStringValue(value_)             \
  ASSERT_EQ_BINARY(                             \
    tokenType::string,                          \
    getTokenType()                              \
  );                                            \
  ASSERT_EQ(                                    \
    value_,                                     \
    token->to<stringToken>().value              \
  )

#define testPrimitiveToken(type_, value_)       \
  getToken();                                   \
  ASSERT_EQ_BINARY(                             \
    tokenType::primitive,                       \
    getTokenType());                            \
  ASSERT_EQ(                                    \
    value_,                                     \
    (type_) token->to<primitiveToken>().value   \
  )
//======================================

#endif
