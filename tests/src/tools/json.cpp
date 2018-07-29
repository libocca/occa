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
#include <sstream>

#include <occa/io.hpp>
#include <occa/tools/json.hpp>
#include <occa/tools/string.hpp>
#include <occa/tools/testing.hpp>

void testAsTypes();
void testString();
void testNumber();
void testObject();
void testArray();
void testKeywords();
void testMethods();
void testSize();
void testTruthyValues();
void testComparisons();
void testConversions();

int main(const int argc, const char **argv) {
  testAsTypes();
  testString();
  testNumber();
  testObject();
  testArray();
  testKeywords();
  testMethods();
  testSize();
  testTruthyValues();
  testComparisons();
  testConversions();

  return 0;
}

void testAsTypes() {
  occa::json j;

  ASSERT_TRUE(j.asString().isString());
  ASSERT_TRUE(j.asNumber().isNumber());
  ASSERT_TRUE(j.asObject().isObject());
  ASSERT_TRUE(j.asArray().isArray());
  ASSERT_TRUE(j.asBoolean().isBoolean());
  ASSERT_TRUE(j.asNull().isNull());

  ASSERT_TRUE(j.asNumber().asString().isString());
  ASSERT_TRUE(j.asObject().asNumber().isNumber());
  ASSERT_TRUE(j.asArray().asObject().isObject());
  ASSERT_TRUE(j.asBoolean().asArray().isArray());
  ASSERT_TRUE(j.asNull().asBoolean().isBoolean());
  ASSERT_TRUE(j.asNumber().asNull().isNull());
}

void testString() {
#define checkString(str_, expected_str_)        \
  j.load(str_);                                 \
  ASSERT_TRUE(j.isString());                    \
  ASSERT_EQ(expected_str_,                      \
            j.string())

  occa::json j;

  // Normal strings
  checkString("\"A\"",
              "A");
  checkString("'A'",
              "A");
  checkString("\"A'\"",
              "A'");
  checkString("'A\"'",
              "A\"");

  // Special chars
  checkString("\"\\\"\"",
              "\"");
  checkString("\"\\\\\"",
              "\\");
  checkString("\"\\/\"",
              "/");
  checkString("\"\\b\"",
              "\b");
  checkString("\"\\f\"",
              "\f");
  checkString("\"\\n\"",
              "\n");
  checkString("\"\\r\"",
              "\r");
  checkString("\"\\t\"",
              "\t");

  // Escape newline
  checkString("\"A\\\nB\"",
              "AB");

  // Test unicode
  checkString("\"\\u0123 \\u4567 \\u89AB \\uCDEF\"",
              "\\u0123 \\u4567 \\u89AB \\uCDEF");
  checkString("\"\\u0123 \\u4567 \\u89ab \\ucdef\"",
              "\\u0123 \\u4567 \\u89ab \\ucdef");

#undef checkString
}

void testNumber() {
#define checkNumber(str_, type_, expected_number_)  \
  j.load(str_);                                     \
  ASSERT_TRUE(j.isNumber());                        \
  ASSERT_EQ(expected_number_,                       \
            (type_) j.number())

  occa::json j;

  checkNumber("-10",
              int, -10);

  checkNumber("10",
              int, 10);

  checkNumber("0.1",
              double, 0.1);

  checkNumber("0.1e10",
              double, 0.1e10);

  checkNumber("0.1E10",
              double, 0.1E10);

  checkNumber("0.1e-10",
              double, 0.1e-10);

  checkNumber("0.1E-10",
              double, 0.1E-10);

  checkNumber("0.1e+10",
              double, 0.1e+10);

  checkNumber("0.1E+10",
              double, 0.1E+10);

#undef checkNumber
}

void testObject() {
#define loadObject(str_, expected_size_)        \
  j.load(str_);                                 \
  ASSERT_TRUE(j.isObject());                    \
  ASSERT_EQ(expected_size_,                     \
            (int) j.object().size())

#define checkNumber(key_, type_, expected_number_)  \
  ASSERT_TRUE(j.object()[key_].isNumber());         \
  ASSERT_EQ(expected_number_,                       \
            (type_) j.object()[key_].number())

  occa::json j;

  loadObject("{\"0\":0, \"1\":1}", 2);
  checkNumber("0", int, 0);
  checkNumber("1", int, 1);

  loadObject("{\"0\":0, \"1\":1,}", 2);
  checkNumber("0", int, 0);
  checkNumber("1", int, 1);

  // Short-hand notation
  loadObject("{0:0, 1:1}", 2);
  checkNumber("0", int, 0);
  checkNumber("1", int, 1);

  loadObject("{0:0, 1:1,}", 2);
  checkNumber("0", int, 0);
  checkNumber("1", int, 1);

  // Test path
  loadObject("{0: {1: {2: {3: 3}}}}", 1);
  ASSERT_EQ(3, (int) j["0/1/2/3"]);

#undef loadObject
#undef checkNumber
}

void testArray() {
#define loadArray(str_, expected_size_)         \
  j.load(str_);                                 \
  ASSERT_TRUE(j.isArray());                     \
  ASSERT_EQ(expected_size_,                     \
            (int) j.array().size())

#define checkNumber(index_, type_, expected_number_)  \
  ASSERT_TRUE(j.array()[index_].isNumber());          \
  ASSERT_EQ(expected_number_,                         \
            (type_) j.array()[index_].number())

  occa::json j;

  loadArray("[1, 2]", 2);
  checkNumber(0, int, 1);
  checkNumber(1, int, 2);

  loadArray("[1, 2,]", 2);
  checkNumber(0, int, 1);
  checkNumber(1, int, 2);

#undef loadArray
#undef checkNumber
}

void testKeywords() {
  occa::json j;

  j.load("true");
  ASSERT_TRUE(j.isBoolean());
  ASSERT_EQ(true,
            j.value_.boolean);

  j.load("false");
  ASSERT_TRUE(j.isBoolean());
  ASSERT_EQ(false,
            j.value_.boolean);

  j.load("null");
  ASSERT_TRUE(j.isNull());
}

void testMethods() {
  occa::json j;

  // Initialize
  ASSERT_FALSE(j.isInitialized());
  j = occa::json::parse("1");
  ASSERT_TRUE(j.isInitialized());

  // Key method
  j.load("{ a: 1, b: 2 }");
  occa::strVector keys = j.keys();
  ASSERT_EQ(2, (int) keys.size());
  ASSERT_EQ("a", keys[0]);
  ASSERT_EQ("b", keys[1]);

  // operator +=
  j.load("1");
  j += 10;
  ASSERT_EQ((int) 11,
            (int) j);

  j.load("'1'");
  j += "1";
  ASSERT_EQ("11",
            j.string());

  j.load("false");
  j += true;
  ASSERT_EQ(true,
            (bool) j);

  // operator += without value
  j = occa::json();
  j += 10;
  ASSERT_EQ((int) 10,
            (int) j);

  j = occa::json();
  j += "1";
  ASSERT_EQ("1",
            j.string());

  j = occa::json();
  j += true;
  ASSERT_EQ(true,
            (bool) j);

  // Default get
  j = occa::json();
  ASSERT_FALSE(j["hi"].isInitialized());
}

void testSize() {
  occa::json j = occa::json::parse(
    "{"
    "  string: 'string',"
    "  number: 1,"
    "  object: { a:0, b:1 },"
    "  array: [0, 1, 2],"
    "  boolean: true,"
    "  null: null,"
    "}"
  );

  ASSERT_EQ(0,
            j["none"].size());
  ASSERT_EQ(0,
            j["null"].size());
  ASSERT_EQ(0,
            j["boolean"].size());
  ASSERT_EQ(0,
            j["number"].size());
  ASSERT_EQ(6,
            j["string"].size());
  ASSERT_EQ(3,
            j["array"].size());
  ASSERT_EQ(2,
            j["object"].size());
}

void testTruthyValues() {
  ASSERT_FALSE(occa::json());

  ASSERT_FALSE(occa::json::parse("false"));
  ASSERT_TRUE(occa::json::parse("true"));

  ASSERT_FALSE(occa::json::parse("0"));
  ASSERT_TRUE(occa::json::parse("1"));

  ASSERT_FALSE(occa::json::parse("''"));
  ASSERT_TRUE(occa::json::parse("'1'"));

  ASSERT_TRUE(occa::json::parse("[]"));
  ASSERT_TRUE(occa::json::parse("[1]"));

  ASSERT_TRUE(occa::json::parse("{}"));
  ASSERT_TRUE(occa::json::parse("{ a: 1 }"));
}

void testComparisons() {
  occa::json data = occa::json::parse(
    "{"
    "  zero: 0,"
    "  one: 1,"
    "  two: 2.0,"
    "  false: false,"
    "  true: true,"
    "  null: null,"
    "  array: [1, 2],"
    "  object: { a: 1, b: 2},"
    "}"
  );
  occa::strVector keys = data.keys();
  for (int j = 0; j < (int) keys.size(); ++j) {
    const std::string jKey = keys[j];
    for (int i = 0; i < (int) keys.size(); ++i) {
      const std::string iKey = keys[i];
      ASSERT_EQ((data[jKey] == data[iKey]),
                (i == j));
    }
  }
}

void testConversions() {
  occa::json j = occa::json::parse(
    "{"
    "  zero: 0,"
    "  one: 1,"
    "  two: 2.0,"
    "  false: false,"
    "  true: true,"
    "  null: null,"
    "}"
  );

  // Test 0
  ASSERT_EQ((bool) false,
            (bool) j["zero"]);

  ASSERT_EQ((int) 0,
            (int) j["zero"]);

  ASSERT_EQ((float) 0,
            (float) j["zero"]);

  ASSERT_EQ((double) 0,
            (double) j["zero"]);

  // Test 1
  ASSERT_EQ((bool) true,
            (bool) j["one"]);

  ASSERT_EQ((int) 1,
            (int) j["one"]);

  ASSERT_EQ((float) 1,
            (float) j["one"]);

  ASSERT_EQ((double) 1,
            (double) j["one"]);

  // Test 2.0
  ASSERT_EQ((bool) true,
            (bool) j["two"]);

  ASSERT_EQ((int) 2,
            (int) j["two"]);

  ASSERT_EQ((float) 2,
            (float) j["two"]);

  ASSERT_EQ((double) 2,
            (double) j["two"]);

  // Test true
  ASSERT_EQ((bool) true,
            (bool) j["true"]);

  ASSERT_EQ((int) 1,
            (int) j["true"]);

  ASSERT_EQ((float) 1,
            (float) j["true"]);

  ASSERT_EQ((double) 1,
            (double) j["true"]);

  // Test false
  ASSERT_EQ((bool) false,
            (bool) j["false"]);

  ASSERT_EQ((int) 0,
            (int) j["false"]);

  ASSERT_EQ((float) 0,
            (float) j["false"]);

  ASSERT_EQ((double) 0,
            (double) j["false"]);

  // Test null
  ASSERT_EQ((bool) false,
            (bool) j["null"]);

  ASSERT_EQ((int) 0,
            (int) j["null"]);

  ASSERT_EQ((float) 0,
            (float) j["null"]);

  ASSERT_EQ((double) 0,
            (double) j["null"]);
}
