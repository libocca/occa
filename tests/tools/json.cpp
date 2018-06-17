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

void testString();
void testNumber();
void testObject();
void testArray();
void testKeywords();
void testMethods();

int main(const int argc, const char **argv) {
  testString();
  testNumber();
  testObject();
  testArray();
  testKeywords();
  testMethods();
  return 0;
}

void testString() {
  occa::json j;

  // Normal strings
  j.load("\"A\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("A", j.value_.string);
  j.load("'A'");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("A", j.value_.string);
  j.load("\"A'\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("A'", j.value_.string);
  j.load("'A\"'");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("A\"", j.value_.string);

  // Special chars
  j.load("\"\\\"\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("\"", j.value_.string);
  j.load("\"\\\\\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("\\", j.value_.string);
  j.load("\"\\/\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("/", j.value_.string);
  j.load("\"\\b\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("\b", j.value_.string);
  j.load("\"\\f\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("\f", j.value_.string);
  j.load("\"\\n\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("\n", j.value_.string);
  j.load("\"\\r\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("\r", j.value_.string);
  j.load("\"\\t\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("\t", j.value_.string);

  // Escape newline
  j.load("\"A\\\nB\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("AB", j.value_.string);

  // Test unicode
  j.load("\"\\u0123 \\u4567 \\u89AB \\uCDEF\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("\\u0123 \\u4567 \\u89AB \\uCDEF",
            j.value_.string);

  j.load("\"\\u0123 \\u4567 \\u89ab \\ucdef\"");
  ASSERT_EQ(occa::json::string_, j.type);
  ASSERT_EQ("\\u0123 \\u4567 \\u89ab \\ucdef",
            j.value_.string);
}

void testNumber() {
  occa::json j;

  j.load("-10");
  ASSERT_EQ(occa::json::number_, j.type);
  ASSERT_EQ(-10, (int) j.value_.number);
  j.load("10");
  ASSERT_EQ(occa::json::number_, j.type);
  ASSERT_EQ(10, (int) j.value_.number);
  j.load("0.1");
  ASSERT_EQ(occa::json::number_, j.type);
  ASSERT_EQ(0.1, (double) j.value_.number);
  j.load("0.1e10");
  ASSERT_EQ(occa::json::number_, j.type);
  ASSERT_EQ(0.1e10, (double) j.value_.number);
  j.load("0.1E10");
  ASSERT_EQ(occa::json::number_, j.type);
  ASSERT_EQ(0.1E10, (double) j.value_.number);
  j.load("0.1e-10");
  ASSERT_EQ(occa::json::number_, j.type);
  ASSERT_EQ(0.1e-10, (double) j.value_.number);
  j.load("0.1E-10");
  ASSERT_EQ(occa::json::number_, j.type);
  ASSERT_EQ(0.1E-10, (double) j.value_.number);
  j.load("0.1e+10");
  ASSERT_EQ(occa::json::number_, j.type);
  ASSERT_EQ(0.1e+10, (double) j.value_.number);
  j.load("0.1E+10");
  ASSERT_EQ(occa::json::number_, j.type);
  ASSERT_EQ(0.1E+10, (double) j.value_.number);
}

void testObject() {
  occa::json j;

  j.load("{\"0\":0, \"1\":1}");
  ASSERT_EQ(occa::json::object_, j.type);
  ASSERT_EQ(2, (int) j.value_.object.size());
  ASSERT_EQ(0, (int) j.value_.object["0"]);
  ASSERT_EQ(1, (int) j.value_.object["1"]);

  j.load("{\"0\":0, \"1\":1,}");
  ASSERT_EQ(occa::json::object_, j.type);
  ASSERT_EQ(2, (int) j.value_.object.size());
  ASSERT_EQ(occa::json::number_, j.value_.object["0"].type);
  ASSERT_EQ(occa::json::number_, j.value_.object["1"].type);
  ASSERT_EQ(0, (int) j.value_.object["0"]);
  ASSERT_EQ(1, (int) j.value_.object["1"]);

  // Short-hand notation
  j.load("{0:0, 1:1}");
  ASSERT_EQ(occa::json::object_, j.type);
  ASSERT_EQ(2, (int) j.value_.object.size());
  ASSERT_EQ(0, (int) j.value_.object["0"]);
  ASSERT_EQ(1, (int) j.value_.object["1"]);

  j.load("{0:0, 1:1,}");
  ASSERT_EQ(occa::json::object_, j.type);
  ASSERT_EQ(2, (int) j.value_.object.size());
  ASSERT_EQ(occa::json::number_, j.value_.object["0"].type);
  ASSERT_EQ(occa::json::number_, j.value_.object["1"].type);
  ASSERT_EQ(0, (int) j.value_.object["0"]);
  ASSERT_EQ(1, (int) j.value_.object["1"]);

  // Test path
  j.load("{0: {1: {2: {3: 3}}}}");
  ASSERT_EQ(3, (int) j["0/1/2/3"]);
}

void testArray() {
  occa::json j;

  j.load("[1, 2]");
  ASSERT_EQ(occa::json::array_, j.type);
  ASSERT_EQ(2, (int) j.value_.array.size());

  ASSERT_EQ(occa::json::number_, j.value_.array[0].type);
  ASSERT_EQ(occa::json::number_, j.value_.array[1].type);
  ASSERT_EQ(1, (int) j.value_.array[0]);
  ASSERT_EQ(2, (int) j.value_.array[1]);

  j.load("[1, 2,]");
  ASSERT_EQ(occa::json::array_, j.type);
  ASSERT_EQ(2, (int) j.value_.array.size());

  ASSERT_EQ(occa::json::number_, j.value_.array[0].type);
  ASSERT_EQ(occa::json::number_, j.value_.array[1].type);
  ASSERT_EQ(1, (int) j.value_.array[0]);
  ASSERT_EQ(2, (int) j.value_.array[1]);
}

void testKeywords() {
  occa::json j;

  j.load("true");
  ASSERT_EQ(occa::json::boolean_, j.type);
  ASSERT_EQ(true, j.value_.boolean);
  j.load("false");
  ASSERT_EQ(occa::json::boolean_, j.type);
  ASSERT_EQ(false, j.value_.boolean);
  j.load("null");
  ASSERT_EQ(occa::json::null_, j.type);
}

void testMethods() {
  occa::json j;

  j.load("{ a: 1, b: 2 }");
  occa::strVector keys = j.keys();
  ASSERT_EQ(2, (int) keys.size());
  ASSERT_EQ("a", keys[0]);
  ASSERT_EQ("b", keys[1]);
}
