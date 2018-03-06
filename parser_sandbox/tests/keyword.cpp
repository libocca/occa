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
#include "keyword.hpp"
#include "typeBuiltins.hpp"

using namespace occa::lang;

void testBuiltins(keywordTrie &keywords);
void testStatements(keywordTrie &keywords);

int main(const int argc, const char **argv) {
  keywordTrie keywords;
  getKeywords(keywords);

  testBuiltins(keywords);
  testStatements(keywords);

  freeKeywords(keywords);

  return 0;
}

void testBuiltins(keywordTrie &keywords) {
  OCCA_ASSERT_EQUAL(keywords.get("const").value().ptr,
                    (void*) &const_);
  OCCA_ASSERT_EQUAL(keywords.get("constexpr").value().ptr,
                    (void*) &constexpr_);
  OCCA_ASSERT_EQUAL(keywords.get("friend").value().ptr,
                    (void*) &friend_);
  OCCA_ASSERT_EQUAL(keywords.get("typedef").value().ptr,
                    (void*) &typedef_);
  OCCA_ASSERT_EQUAL(keywords.get("signed").value().ptr,
                    (void*) &signed_);
  OCCA_ASSERT_EQUAL(keywords.get("unsigned").value().ptr,
                    (void*) &unsigned_);
  OCCA_ASSERT_EQUAL(keywords.get("volatile").value().ptr,
                    (void*) &volatile_);

  OCCA_ASSERT_EQUAL(keywords.get("extern").value().ptr,
                    (void*) &extern_);
  OCCA_ASSERT_EQUAL(keywords.get("mutable").value().ptr,
                    (void*) &mutable_);
  OCCA_ASSERT_EQUAL(keywords.get("register").value().ptr,
                    (void*) &register_);
  OCCA_ASSERT_EQUAL(keywords.get("static").value().ptr,
                    (void*) &static_);
  OCCA_ASSERT_EQUAL(keywords.get("thread_local").value().ptr,
                    (void*) &thread_local_);

  OCCA_ASSERT_EQUAL(keywords.get("explicit").value().ptr,
                    (void*) &explicit_);
  OCCA_ASSERT_EQUAL(keywords.get("inline").value().ptr,
                    (void*) &inline_);
  OCCA_ASSERT_EQUAL(keywords.get("virtual").value().ptr,
                    (void*) &virtual_);

  OCCA_ASSERT_EQUAL(keywords.get("class").value().ptr,
                    (void*) &class_);
  OCCA_ASSERT_EQUAL(keywords.get("enum").value().ptr,
                    (void*) &enum_);
  OCCA_ASSERT_EQUAL(keywords.get("struct").value().ptr,
                    (void*) &struct_);
  OCCA_ASSERT_EQUAL(keywords.get("union").value().ptr,
                    (void*) &union_);

  OCCA_ASSERT_EQUAL(keywords.get("bool").value().ptr,
                    (void*) &bool_);
  OCCA_ASSERT_EQUAL(keywords.get("char").value().ptr,
                    (void*) &char_);
  OCCA_ASSERT_EQUAL(keywords.get("char16_t").value().ptr,
                    (void*) &char16_t_);
  OCCA_ASSERT_EQUAL(keywords.get("char32_t").value().ptr,
                    (void*) &char32_t_);
  OCCA_ASSERT_EQUAL(keywords.get("wchar_t").value().ptr,
                    (void*) &wchar_t_);
  OCCA_ASSERT_EQUAL(keywords.get("short").value().ptr,
                    (void*) &short_);
  OCCA_ASSERT_EQUAL(keywords.get("int").value().ptr,
                    (void*) &int_);
  OCCA_ASSERT_EQUAL(keywords.get("long").value().ptr,
                    (void*) &long_);
  OCCA_ASSERT_EQUAL(keywords.get("float").value().ptr,
                    (void*) &float_);
  OCCA_ASSERT_EQUAL(keywords.get("double").value().ptr,
                    (void*) &double_);
  OCCA_ASSERT_EQUAL(keywords.get("void").value().ptr,
                    (void*) &void_);
  OCCA_ASSERT_EQUAL(keywords.get("auto").value().ptr,
                    (void*) &auto_);
}

void testStatements(keywordTrie &keywords) {
  OCCA_ASSERT_TRUE(keywords.has("if"));
  OCCA_ASSERT_TRUE(keywords.has("else"));
  OCCA_ASSERT_TRUE(keywords.has("switch"));
  OCCA_ASSERT_TRUE(keywords.has("case"));
  OCCA_ASSERT_TRUE(keywords.has("default"));
  OCCA_ASSERT_TRUE(keywords.has("for"));
  OCCA_ASSERT_TRUE(keywords.has("while"));
  OCCA_ASSERT_TRUE(keywords.has("do"));
  OCCA_ASSERT_TRUE(keywords.has("break"));
  OCCA_ASSERT_TRUE(keywords.has("continue"));
  OCCA_ASSERT_TRUE(keywords.has("return"));
  OCCA_ASSERT_TRUE(keywords.has("goto"));
}
