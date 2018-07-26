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
#include <occa/lang/keyword.hpp>

using namespace occa::lang;

void testDefaults(keywordMap &keywords);

int main(const int argc, const char **argv) {
  keywordMap keywords;
  getKeywords(keywords);

  testDefaults(keywords);

  freeKeywords(keywords);

  return 0;
}

#define assertKeyword(name_, type_)             \
  ASSERT_EQ_BINARY(type_,                       \
                   keywords[name_]->type())


void testDefaults(keywordMap &keywords) {
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
