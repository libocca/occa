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

#include "occa/tools/io.hpp"
#include "occa/tools/json.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/testing.hpp"

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
}

void testString() {
  occa::json j;

  // Normal strings
  j.load("\"A\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("A", j.value_.string);
  j.load("'A'");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("A", j.value_.string);
  j.load("\"A'\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("A'", j.value_.string);
  j.load("'A\"'");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("A\"", j.value_.string);

  // Special chars
  j.load("\"\\\"\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("\"", j.value_.string);
  j.load("\"\\\\\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("\\", j.value_.string);
  j.load("\"\\/\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("/", j.value_.string);
  j.load("\"\\b\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("\b", j.value_.string);
  j.load("\"\\f\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("\f", j.value_.string);
  j.load("\"\\n\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("\n", j.value_.string);
  j.load("\"\\r\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("\r", j.value_.string);
  j.load("\"\\t\"");
  OCCA_ASSERT_EQUAL(occa::json::string_, j.type);
  OCCA_ASSERT_EQUAL("\t", j.value_.string);
}

void testNumber() {
  occa::json j;

  j.load("-10");
  OCCA_ASSERT_EQUAL(occa::json::number_, j.type);
  OCCA_ASSERT_EQUAL(-10, (int) j.value_.number);
  j.load("10");
  OCCA_ASSERT_EQUAL(occa::json::number_, j.type);
  OCCA_ASSERT_EQUAL(10, (int) j.value_.number);
  j.load("0.1");
  OCCA_ASSERT_EQUAL(occa::json::number_, j.type);
  OCCA_ASSERT_EQUAL(0.1, (double) j.value_.number);
  j.load("0.1e10");
  OCCA_ASSERT_EQUAL(occa::json::number_, j.type);
  OCCA_ASSERT_EQUAL(0.1e10, (double) j.value_.number);
  j.load("0.1E10");
  OCCA_ASSERT_EQUAL(occa::json::number_, j.type);
  OCCA_ASSERT_EQUAL(0.1E10, (double) j.value_.number);
  j.load("0.1e-10");
  OCCA_ASSERT_EQUAL(occa::json::number_, j.type);
  OCCA_ASSERT_EQUAL(0.1e-10, (double) j.value_.number);
  j.load("0.1E-10");
  OCCA_ASSERT_EQUAL(occa::json::number_, j.type);
  OCCA_ASSERT_EQUAL(0.1E-10, (double) j.value_.number);
  j.load("0.1e+10");
  OCCA_ASSERT_EQUAL(occa::json::number_, j.type);
  OCCA_ASSERT_EQUAL(0.1e+10, (double) j.value_.number);
  j.load("0.1E+10");
  OCCA_ASSERT_EQUAL(occa::json::number_, j.type);
  OCCA_ASSERT_EQUAL(0.1E+10, (double) j.value_.number);
}

void testObject() {
  occa::json j;

  j.load("{\"0\":0, \"1\":1}");
  OCCA_ASSERT_EQUAL(occa::json::object_, j.type);
  OCCA_ASSERT_EQUAL(2, (int) j.value_.object.size());
  OCCA_ASSERT_EQUAL(0, (int) j.value_.object["0"]);
  OCCA_ASSERT_EQUAL(1, (int) j.value_.object["1"]);

  j.load("{\"0\":0, \"1\":1,}");
  OCCA_ASSERT_EQUAL(occa::json::object_, j.type);
  OCCA_ASSERT_EQUAL(2, (int) j.value_.object.size());
  OCCA_ASSERT_EQUAL(occa::json::number_, j.value_.object["0"].type);
  OCCA_ASSERT_EQUAL(occa::json::number_, j.value_.object["1"].type);
  OCCA_ASSERT_EQUAL(0, (int) j.value_.object["0"]);
  OCCA_ASSERT_EQUAL(1, (int) j.value_.object["1"]);

  // Short-hand notation
  j.load("{0:0, 1:1}");
  OCCA_ASSERT_EQUAL(occa::json::object_, j.type);
  OCCA_ASSERT_EQUAL(2, (int) j.value_.object.size());
  OCCA_ASSERT_EQUAL(0, (int) j.value_.object["0"]);
  OCCA_ASSERT_EQUAL(1, (int) j.value_.object["1"]);

  j.load("{0:0, 1:1,}");
  OCCA_ASSERT_EQUAL(occa::json::object_, j.type);
  OCCA_ASSERT_EQUAL(2, (int) j.value_.object.size());
  OCCA_ASSERT_EQUAL(occa::json::number_, j.value_.object["0"].type);
  OCCA_ASSERT_EQUAL(occa::json::number_, j.value_.object["1"].type);
  OCCA_ASSERT_EQUAL(0, (int) j.value_.object["0"]);
  OCCA_ASSERT_EQUAL(1, (int) j.value_.object["1"]);

  // Test path
  j.load("{0: {1: {2: {3: 3}}}}");
  OCCA_ASSERT_EQUAL(3, (int) j["0/1/2/3"]);
}

void testArray() {
  occa::json j;

  j.load("[1, 2]");
  OCCA_ASSERT_EQUAL(occa::json::array_, j.type);
  OCCA_ASSERT_EQUAL(2, (int) j.value_.array.size());

  OCCA_ASSERT_EQUAL(occa::json::number_, j.value_.array[0].type);
  OCCA_ASSERT_EQUAL(occa::json::number_, j.value_.array[1].type);
  OCCA_ASSERT_EQUAL(1, (int) j.value_.array[0]);
  OCCA_ASSERT_EQUAL(2, (int) j.value_.array[1]);

  j.load("[1, 2,]");
  OCCA_ASSERT_EQUAL(occa::json::array_, j.type);
  OCCA_ASSERT_EQUAL(2, (int) j.value_.array.size());

  OCCA_ASSERT_EQUAL(occa::json::number_, j.value_.array[0].type);
  OCCA_ASSERT_EQUAL(occa::json::number_, j.value_.array[1].type);
  OCCA_ASSERT_EQUAL(1, (int) j.value_.array[0]);
  OCCA_ASSERT_EQUAL(2, (int) j.value_.array[1]);
}

void testKeywords() {
  occa::json j;

  j.load("true");
  OCCA_ASSERT_EQUAL(occa::json::boolean_, j.type);
  OCCA_ASSERT_EQUAL(true, j.value_.boolean);
  j.load("false");
  OCCA_ASSERT_EQUAL(occa::json::boolean_, j.type);
  OCCA_ASSERT_EQUAL(false, j.value_.boolean);
  j.load("null");
  OCCA_ASSERT_EQUAL(occa::json::null_, j.type);
}

void testMethods() {
  occa::json j;

  j.load("{ a: 1, b: 2 }");
  occa::strVector keys = j.keys();
  OCCA_ASSERT_EQUAL(2, (int) keys.size());
  OCCA_ASSERT_EQUAL("a", keys[0]);
  OCCA_ASSERT_EQUAL("b", keys[1]);
}
