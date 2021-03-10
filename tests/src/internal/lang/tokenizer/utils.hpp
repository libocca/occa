#ifndef OCCA_TESTS_LANG_TOKENIZER_UTILS
#define OCCA_TESTS_LANG_TOKENIZER_UTILS

#include <occa/internal/utils/testing.hpp>

#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/tokenizer.hpp>
#include <occa/internal/lang/processingStages.hpp>

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
#define testCommentPeek(s)                      \
  setStream(s);                                 \
  ASSERT_EQ_BINARY(tokenType::op,               \
                   tokenizer.peek())

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

#define testCommentToken(s)                     \
  setToken(s);                                  \
  ASSERT_EQ_BINARY(tokenType::comment,          \
                   getTokenType());             \

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
                   getTokenType());                 \
  ASSERT_EQ_BINARY(encoding_,                       \
                   token->to<charToken>().encoding)

#define testCommentValue(s, value_)             \
  setStream(s);                                 \
  getToken();                                   \
  testNextCommentValue(value_)

#define testNextCommentValue(value_)            \
  ASSERT_EQ_BINARY(                             \
    tokenType::comment,                         \
    getTokenType()                              \
  );                                            \
  ASSERT_EQ(                                    \
    value_,                                     \
    token->to<commentToken>().value             \
  )

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
