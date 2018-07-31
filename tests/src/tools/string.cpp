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

using namespace occa;

void testStrip();
void testEscape();
void testSplit();
void testCaseMethods();
void testString();
void testMatchEnds();
void testHex();
void testJoin();
void testColors();

int main(const int argc, const char **argv) {
  testStrip();
  testEscape();
  testSplit();
  testCaseMethods();
  testString();
  testMatchEnds();
  testHex();
  testJoin();
  testColors();

  return 0;
}

void testStrip() {
  ASSERT_EQ(strip("a"), "a");
  ASSERT_EQ(strip("  a"), "a");
  ASSERT_EQ(strip("a  "), "a");
  ASSERT_EQ(strip("  a  "), "a");
}

void testEscape() {
  ASSERT_EQ(escape("a", 'b'), "a");
  ASSERT_EQ(escape("aba", 'b'), "a\\ba");
  ASSERT_EQ(escape("a\\ba", 'b'), "a\\\\ba");
  ASSERT_EQ(escape("a", 'b', '|'), "a");
  ASSERT_EQ(escape("aba", 'b', '|'), "a|ba");

  ASSERT_EQ(unescape("a", 'b'), "a");
  ASSERT_EQ(unescape("a\\ba", 'b'), "aba");
  ASSERT_EQ(unescape("a", 'b', '|'), "a");
  ASSERT_EQ(unescape("a|ba", 'b', '|'), "aba");
}

void testSplit() {
  strVector vec = split("|a|b|c", '|');
  ASSERT_EQ((int) vec.size(), 4);
  ASSERT_IN(std::string(""), vec);
  ASSERT_IN("a", vec);
  ASSERT_IN("b", vec);
  ASSERT_IN("c", vec);
}

void testCaseMethods() {
  // Uppercase
  ASSERT_EQ(uppercase('A'), 'A');
  ASSERT_EQ(uppercase('a'), 'A');
  ASSERT_EQ(uppercase(','), ',');

  ASSERT_EQ(uppercase("abc123"), "ABC123");
  ASSERT_EQ(uppercase("abc123", 3), "ABC");

  // Lowercase
  ASSERT_EQ(lowercase('A'), 'a');
  ASSERT_EQ(lowercase('a'), 'a');
  ASSERT_EQ(lowercase(','), ',');

  ASSERT_EQ(lowercase("ABC123"), "abc123");
  ASSERT_EQ(lowercase("abc123", 3), "abc");
}

void testString() {
  strVector vec = split("a|b|c", '|');

  // toString
  ASSERT_EQ(toString(1), "1");
  ASSERT_EQ(toString("1"), "1");
  ASSERT_EQ(toString(vec), "[a,b,c]");

  // fromString
  ASSERT_EQ(fromString<int>("1"), 1);
  ASSERT_TRUE(listFromString<std::string>(toString(vec)) == vec);
}

void testMatchEnds() {
  ASSERT_TRUE(startsWith("abc", "a"));
  ASSERT_TRUE(startsWith("abc", "ab"));
  ASSERT_FALSE(startsWith("abc", "ab_"));

  ASSERT_TRUE(endsWith("abc", "c"));
  ASSERT_TRUE(endsWith("abc", "bc"));
  ASSERT_FALSE(endsWith("abc", "_bc"));
}

void testHex() {
  // To Hex
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(toHexChar(i), '0' + i);
  }
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(toHexChar(10 + i), 'a' + i);
  }

  // From Hex
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(fromHexChar('0' + i), i);
  }
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(fromHexChar('a' + i), 10 + i);
  }

  ASSERT_EQ(toHex((char) 0x01), "01");
  ASSERT_EQ(toHex((char) 0x0A), "0a");
  ASSERT_EQ(toHex((char) 0xAF), "af");

  ASSERT_EQ(fromHex<char>("01"), (char) 0x01);
  ASSERT_EQ(fromHex<char>("0A"), (char) 0x0A);
  ASSERT_EQ(fromHex<char>("AF"), (char) 0xAF);

  // Stringify bits
  ASSERT_EQ(stringifySetBits(0),
            "No bits set");
  ASSERT_EQ(stringifySetBits(0x01),
            "0");
  ASSERT_EQ(stringifySetBits(0x05),
            "0, 2");
  ASSERT_EQ(stringifySetBits(0x0F),
            "0, 1, 2, 3");

  ASSERT_EQ(stringifyBytes(1L << 2),
            "4 bytes");
  ASSERT_EQ(stringifyBytes(1L << 12),
            "4 KB");
  ASSERT_EQ(stringifyBytes(1L << 22),
            "4 MB");
  ASSERT_EQ(stringifyBytes(1L << 32),
            "4 GB");
  ASSERT_EQ(stringifyBytes(1L << 42),
            "4 TB");
  ASSERT_EQ(stringifyBytes(1L << 52),
            toString(1L << 52) + " bytes");
}

void testJoin() {
  strVector vec;
  vec.push_back("a");
  vec.push_back("b");
  vec.push_back("c");
  ASSERT_EQ(join(vec, ","),
            "a,b,c");
  ASSERT_EQ(join(vec, " , "),
            "a , b , c");
}

void testColors() {
  std::cout << black("black") << '\n'
            << red("red") << '\n'
            << green("green") << '\n'
            << yellow("yellow") << '\n'
            << blue("blue") << '\n'
            << magenta("magenta") << '\n'
            << cyan("cyan") << '\n'
            << white("white") << '\n';
}
