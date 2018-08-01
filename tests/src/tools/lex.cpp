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
#include <occa.hpp>
#include <occa/tools/testing.hpp>

void testCharsetMethods();
void testSkipToMethods();
void testSkipFromMethods();
void testWhitespaceMethods();

int main(const int argc, const char **argv) {
  testCharsetMethods();
  testSkipToMethods();
  testSkipFromMethods();
  testWhitespaceMethods();

  return 0;
}

void testCharsetMethods() {
  for (int i = 0; i < 10; ++i) {
    char c = '0' + i;
    ASSERT_TRUE(occa::lex::isDigit(c));
    ASSERT_FALSE(occa::lex::isAlpha(c));
    ASSERT_TRUE(occa::lex::inCharset(c, occa::lex::numberCharset));
  }
  for (int i = 0; i < 26; ++i) {
    char c = 'a' + i;
    ASSERT_FALSE(occa::lex::isDigit(c));
    ASSERT_TRUE(occa::lex::isAlpha(c));
    ASSERT_FALSE(occa::lex::inCharset(c, occa::lex::numberCharset));

    c = 'A' + i;
    ASSERT_FALSE(occa::lex::isDigit(c));
    ASSERT_TRUE(occa::lex::isAlpha(c));
    ASSERT_FALSE(occa::lex::inCharset(c, occa::lex::numberCharset));
  }

  ASSERT_TRUE(occa::lex::inCharset('a', "abc"));
  ASSERT_FALSE(occa::lex::inCharset('d', "abc"));
}

void testSkipToMethods() {
  const char *c0, *c;
  char escapeChar = '.';
  char delimiter = '|';
  std::string delimiters = "|><";

  std::string s1 = "abc|";        // skipTo(|)
  std::string s2 = "abc.|def|";   // skipTo(|, delimiter=.)
  std::string s3 = "abc<";        // skipTo(<>|)
  std::string s4 = "abc.<.>def|"; // skipTo(<>|, delimiter=.)

  c0 = c = s1.c_str();
  occa::lex::skipTo(c, delimiter);
  ASSERT_EQ(c, c0 + s1.size() - 1);
  ++c;
  occa::lex::skipTo(c, delimiter);
  ASSERT_EQ(*c, '\0');

  c0 = c = s2.c_str();
  occa::lex::skipTo(c, delimiter, escapeChar);
  ASSERT_EQ(c, c0 + s2.size() - 1);
  ++c;
  occa::lex::skipTo(c, delimiter, escapeChar);
  ASSERT_EQ(*c, '\0');

  c0 = c = s3.c_str();
  occa::lex::skipTo(c, delimiters.c_str());
  ASSERT_EQ(c, c0 + s3.size() - 1);
  ++c;
  occa::lex::skipTo(c, delimiters.c_str());
  ASSERT_EQ(*c, '\0');

  c0 = c = s4.c_str();
  occa::lex::skipTo(c, delimiters.c_str(), escapeChar);
  ASSERT_EQ(c, c0 + s4.size() - 1);
  ++c;
  occa::lex::skipTo(c, delimiters.c_str(), escapeChar);
  ASSERT_EQ(*c, '\0');

  // Combined
  std::string content = s1 + s2 + s3 + s4;

  c0 = c = content.c_str();
  occa::lex::skipTo(c, delimiter);
  ASSERT_EQ(c, c0 + s1.size() - 1);

  c0 = ++c;
  occa::lex::skipTo(c, delimiter, escapeChar);
  ASSERT_EQ(c, c0 + s2.size() - 1);

  c0 = ++c;
  occa::lex::skipTo(c, delimiters.c_str());
  ASSERT_EQ(c, c0 + s3.size() - 1);

  c0 = ++c;
  occa::lex::skipTo(c, delimiters.c_str(), escapeChar);
  ASSERT_EQ(c, c0 + s4.size() - 1);
}

void testSkipFromMethods() {
  const char *c0, *c;
  std::string delimiters = "cba";

  std::string content = "|abc|";
  c0 = c = content.c_str();

  occa::lex::skipFrom(c, delimiters.c_str());
  ASSERT_EQ(c - c0, 0);

  c0 = ++c;
  occa::lex::skipFrom(c, delimiters.c_str());
  ASSERT_EQ(c - c0, 3);

  c0 = ++c;
  occa::lex::skipFrom(c, delimiters.c_str());
  ASSERT_EQ(*c, '\0');
}

void testWhitespaceMethods() {
  const char *c0, *c;
  std::string content = "|  \t\n ||||||    |";
  c0 = c = content.c_str();

  // "|" -> "|"
  occa::lex::skipWhitespace(c);
  ASSERT_EQ(c - c0, 0);

  // "|" -> "|  \t\n |"
  c0 = ++c;
  occa::lex::skipWhitespace(c);
  ASSERT_GT(c - c0, 0);
  ASSERT_EQ(*c, '|');

  // "|  \t\n |" -> "|  \t\n |"
  c0 = c;
  occa::lex::skipWhitespace(c);
  ASSERT_EQ(c - c0, 0);

  // "|  \t\n |" -> "|  \t\n |||||| "
  occa::lex::skipToWhitespace(c);
  ASSERT_GT(c - c0, 0);
  ASSERT_EQ(*c, ' ');

  // "|  \t\n |||||| " -> "|  \t\n |||||| "
  c0 = c;
  occa::lex::skipToWhitespace(c);
  ASSERT_EQ(c - c0, 0);

  c0 = c = content.c_str() + content.size();

  // Test \0
  occa::lex::skipWhitespace(c);
  ASSERT_EQ(c - c0, 0);

  occa::lex::skipToWhitespace(c);
  ASSERT_EQ(c - c0, 0);
}
