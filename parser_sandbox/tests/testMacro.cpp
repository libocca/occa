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

#include "occa/tools/env.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/testing.hpp"

#include "preprocessor.hpp"
#include "macro.hpp"
#include "specialMacros.hpp"

void testPlainMacros();
void testFunctionMacros();
void testSpecialMacros();
void testErrors();

int main(const int argc, const char **argv) {
  testPlainMacros();
  testFunctionMacros();
  testSpecialMacros();
  testErrors();
}

void testPlainMacros() {
  occa::preprocessor_t preprocessor;
  preprocessor.exitOnFatalError = false;
  preprocessor.processSource("");
  occa::macro_t macro(&preprocessor, "A");

  OCCA_ASSERT_EQUAL(macro.name, "A");
  OCCA_ASSERT_EQUAL(macro.expand(""), "");

  macro.load("B 1 2 3");
  OCCA_ASSERT_EQUAL(macro.name, "B");
  OCCA_ASSERT_EQUAL(macro.expand(""), "1 2 3");
}

void testFunctionMacros() {
  occa::preprocessor_t preprocessor;
  preprocessor.exitOnFatalError = false;
  preprocessor.processSource("");
  occa::macro_t macro(&preprocessor, "FOO(A) A");

  OCCA_ASSERT_EQUAL("",
                    macro.expand("()"));
  OCCA_ASSERT_EQUAL("1",
                    macro.expand("(1)"));

  macro.load("FOO(A, B) A B");
  OCCA_ASSERT_EQUAL("2 3",
                    macro.expand("(2, 3)"));
  OCCA_ASSERT_EQUAL("4 5",
                    macro.expand("(4, 5, 6)"));

  macro.load("FOO(A, B) A##B");
  OCCA_ASSERT_EQUAL("6",
                    macro.expand("(, 6)"));
  OCCA_ASSERT_EQUAL("07",
                    macro.expand("(0, 7)"));

  macro.load("FOO(A, ...) A __VA_ARGS__");
  OCCA_ASSERT_EQUAL("7 ",
                    macro.expand("(7,)"));
  OCCA_ASSERT_EQUAL("8 9, 10,",
                    macro.expand("(8, 9, 10,)"));

  macro.load("FOO(...) (X, ##__VA_ARGS__)");
  OCCA_ASSERT_EQUAL("(X, 11 )",
                    macro.expand("(11,)"));
  OCCA_ASSERT_EQUAL("(X, )",
                    macro.expand("()"));

  macro.load("FOO(A) #A");
  OCCA_ASSERT_EQUAL("\"12\"",
                    macro.expand("(12)"));

  macro.load("FOO(A, B) #A##B");
  OCCA_ASSERT_EQUAL("\"1\"3",
                    macro.expand("(1, 3)"));
}

void testSpecialMacros() {
  occa::preprocessor_t preprocessor;
  preprocessor.exitOnFatalError = false;
  preprocessor.processSource("#line 10 foo");

  char *c = new char[1];

  occa::fileMacro fileMacro(&preprocessor);       // __FILE__
  occa::lineMacro lineMacro(&preprocessor);       // __LINE__
  occa::counterMacro counterMacro(&preprocessor); // __COUNTER__

  OCCA_ASSERT_EQUAL(occa::env::PWD + "foo",
                    fileMacro.expand(c));

  OCCA_ASSERT_EQUAL("9",
                    lineMacro.expand(c));

  OCCA_ASSERT_EQUAL("0",
                    counterMacro.expand(c));
  OCCA_ASSERT_EQUAL("1",
                    counterMacro.expand(c));
  OCCA_ASSERT_EQUAL("2",
                    counterMacro.expand(c));

  delete [] c;
}

void testErrors() {
  occa::preprocessor_t preprocessor;
  std::stringstream ss;
  preprocessor.exitOnFatalError = false;
  preprocessor.setOutputStream(ss);
  preprocessor.processSource("");

  const bool printOutput = false;

  // Missing closing )
  occa::macro_t macro(&preprocessor, "FOO(A) A");
  macro.expand("(1");

  OCCA_ASSERT_EQUAL(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  // No macro name
  macro.load("");
  OCCA_ASSERT_EQUAL(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  // Identifier starts badly
  macro.load("0A 0");
  OCCA_ASSERT_EQUAL(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  // No whitespace warning
  macro.load("FOO-10");
  OCCA_ASSERT_EQUAL(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  // Variadic in wrong position
  macro.load("FOO(A, ..., B)");
  OCCA_ASSERT_EQUAL(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");
}
