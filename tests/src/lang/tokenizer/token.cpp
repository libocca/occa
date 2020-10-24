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
  setStream("true_case");
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

  testCommentPeek("// hi");
  testCommentPeek("  // hi");
  testCommentPeek("// hi\n bye");

  testCommentPeek("/* hi */");
  testCommentPeek("    /* hi */   ");
  testCommentPeek("/*\n hi \n*/");
  testCommentPeek("/* hi */\nbye");

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

  setStream("<foobar>");
  ASSERT_TRUE(tokenizer.loadingAngleBracketHeader());
  ASSERT_FALSE(tokenizer.loadingQuotedHeader());

  setStream("\"foobar\"");
  ASSERT_FALSE(tokenizer.loadingAngleBracketHeader());
  ASSERT_TRUE(tokenizer.loadingQuotedHeader());

  setStream("foobar");
  ASSERT_FALSE(tokenizer.loadingAngleBracketHeader());
  ASSERT_FALSE(tokenizer.loadingQuotedHeader());

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
  setToken("true_case");
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

  testCommentValue("// hi", "// hi");
  testCommentValue("  // hi", "// hi");
  testCommentValue("// hi\n bye", "// hi");

  testCommentValue("/* hi */", "/* hi */");
  testCommentValue("    /* hi */   ", "/* hi */");
  testCommentValue("/*\n hi \n*/", "/*\n hi \n*/");
  testCommentValue("/* hi */\nbye", "/* hi */");

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
