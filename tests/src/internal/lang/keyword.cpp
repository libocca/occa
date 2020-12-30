#include <occa/internal/utils/testing.hpp>
#include <occa/internal/lang/keyword.hpp>

using namespace occa::lang;

void testDefaults(keywords_t &keywords);

int main(const int argc, const char **argv) {
  keywords_t keywords;
  getKeywords(keywords);

  testDefaults(keywords);

  keywords.free();

  return 0;
}

#define assertKeyword(name_, type_)             \
  ASSERT_EQ_BINARY(type_,                       \
                   keywords.keywords[name_]->type())


void testDefaults(keywords_t &keywords) {
  // Qualifiers
  assertKeyword("const"       , keywordType::qualifier);
  assertKeyword("constexpr"   , keywordType::qualifier);
  assertKeyword("friend"      , keywordType::qualifier);
  assertKeyword("typedef"     , keywordType::qualifier);
  assertKeyword("signed"      , keywordType::qualifier);
  assertKeyword("unsigned"    , keywordType::qualifier);
  assertKeyword("volatile"    , keywordType::qualifier);
  assertKeyword("long"        , keywordType::qualifier);

  assertKeyword("extern"      , keywordType::qualifier);
  assertKeyword("mutable"     , keywordType::qualifier);
  assertKeyword("register"    , keywordType::qualifier);
  assertKeyword("static"      , keywordType::qualifier);
  assertKeyword("thread_local", keywordType::qualifier);

  assertKeyword("explicit"    , keywordType::qualifier);
  assertKeyword("inline"      , keywordType::qualifier);
  assertKeyword("virtual"     , keywordType::qualifier);

  assertKeyword("class"       , keywordType::qualifier);
  assertKeyword("enum"        , keywordType::qualifier);
  assertKeyword("struct"      , keywordType::qualifier);
  assertKeyword("union"       , keywordType::qualifier);

  // Types
  assertKeyword("bool"    , keywordType::type);
  assertKeyword("char"    , keywordType::type);
  assertKeyword("char16_t", keywordType::type);
  assertKeyword("char32_t", keywordType::type);
  assertKeyword("wchar_t" , keywordType::type);
  assertKeyword("short"   , keywordType::type);
  assertKeyword("int"     , keywordType::type);
  assertKeyword("float"   , keywordType::type);
  assertKeyword("double"  , keywordType::type);
  assertKeyword("void"    , keywordType::type);
  assertKeyword("auto"    , keywordType::type);

  // Statements
  assertKeyword("if"      , keywordType::if_);
  assertKeyword("else"    , keywordType::else_);
  assertKeyword("switch"  , keywordType::switch_);
  assertKeyword("case"    , keywordType::case_);
  assertKeyword("default" , keywordType::default_);
  assertKeyword("for"     , keywordType::for_);
  assertKeyword("while"   , keywordType::while_);
  assertKeyword("do"      , keywordType::do_);
  assertKeyword("break"   , keywordType::break_);
  assertKeyword("continue", keywordType::continue_);
  assertKeyword("return"  , keywordType::return_);
  assertKeyword("goto"    , keywordType::goto_);
}
