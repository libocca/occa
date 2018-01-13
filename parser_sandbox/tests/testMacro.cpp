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

  OCCA_TEST_COMPARE(macro.name, "A");
  OCCA_TEST_COMPARE(macro.expand(""), "");

  macro.load("B 1 2 3");
  OCCA_TEST_COMPARE(macro.name, "B");
  OCCA_TEST_COMPARE(macro.expand(""), "1 2 3");
}

void testFunctionMacros() {
  occa::preprocessor_t preprocessor;
  preprocessor.exitOnFatalError = false;
  preprocessor.processSource("");
  occa::macro_t macro(&preprocessor, "FOO(A) A");

  OCCA_TEST_COMPARE("",
                    macro.expand("()"));
  OCCA_TEST_COMPARE("1",
                    macro.expand("(1)"));

  macro.load("FOO(A, B) A B");
  OCCA_TEST_COMPARE("2 3",
                    macro.expand("(2, 3)"));
  OCCA_TEST_COMPARE("4 5",
                    macro.expand("(4, 5, 6)"));

  macro.load("FOO(A, B) A##B");
  OCCA_TEST_COMPARE("6",
                    macro.expand("(, 6)"));
  OCCA_TEST_COMPARE("07",
                    macro.expand("(0, 7)"));

  macro.load("FOO(A, ...) A __VA_ARGS__");
  OCCA_TEST_COMPARE("7 ",
                    macro.expand("(7,)"));
  OCCA_TEST_COMPARE("8 9, 10,",
                    macro.expand("(8, 9, 10,)"));

  macro.load("FOO(...) (X, ##__VA_ARGS__)");
  OCCA_TEST_COMPARE("(X, 11 )",
                    macro.expand("(11,)"));
  OCCA_TEST_COMPARE("(X, )",
                    macro.expand("()"));

  macro.load("FOO(A) #A");
  OCCA_TEST_COMPARE("\"12\"",
                    macro.expand("(12)"));

  macro.load("FOO(A, B) #A##B");
  OCCA_TEST_COMPARE("\"1\"3",
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

  OCCA_TEST_COMPARE(occa::env::PWD + "foo",
                    fileMacro.expand(c));

  OCCA_TEST_COMPARE("9",
                    lineMacro.expand(c));

  OCCA_TEST_COMPARE("0",
                    counterMacro.expand(c));
  OCCA_TEST_COMPARE("1",
                    counterMacro.expand(c));
  OCCA_TEST_COMPARE("2",
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

  OCCA_TEST_COMPARE(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  // No macro name
  macro.load("");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  // Identifier starts badly
  macro.load("0A 0");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  // No whitespace warning
  macro.load("FOO-10");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  // Variadic in wrong position
  macro.load("FOO(A, ..., B)");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");
}
