#include "utils.hpp"

void testExternMerging();
void testPrimitiveMethods();
void testErrors();

int main(const int argc, const char **argv) {
  testExternMerging();
  testPrimitiveMethods();
  testErrors();

  return 0;
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
