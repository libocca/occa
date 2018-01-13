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
#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "occa/tools/testing.hpp"
#include "occa/parser/primitive.hpp"

void testLoad();
void testBadParsing();
void testToString();

int main(const int argc, const char **argv) {
  testLoad();
  testBadParsing();
  testToString();
}

void testLoad() {
  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("15"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-15"));

  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("0xF"));
  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("0XF"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-0xF"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-0XF"));

  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("0b1111"));
  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("0B1111"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-0b1111"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-0B1111"));

  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("15.01"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-15.01"));

  OCCA_TEST_COMPARE(1e-16,
                    (double) occa::primitive("1e-16"));
  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("1.501e1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-1.501e1"));
  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("1.501E1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-1.501E1"));

  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("1.501e+1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-1.501e+1"));
  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("1.501E+1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-1.501E+1"));

  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("150.1e-1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-150.1e-1"));
  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("150.1E-1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-150.1E-1"));
}

void testBadParsing() {
  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive("").type);

  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive(" ").type);

  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive("-").type);

  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive("+").type);

  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive("-   ").type);

  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive("+   ").type);

  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive("0x").type);

  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive("0b").type);

  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive("A").type);

  OCCA_TEST_COMPARE(occa::primitiveType::none,
                    occa::primitive("*").type);
}

void testToString() {
  OCCA_TEST_COMPARE("68719476735L",
                    (std::string) occa::primitive("0xFFFFFFFFF"));

  OCCA_TEST_COMPARE("NaN",
                    (std::string) occa::primitive(""));
}
