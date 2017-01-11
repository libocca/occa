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

  occa::fileMacro_t fileMacro(&preprocessor);       // __FILE__
  occa::lineMacro_t lineMacro(&preprocessor);       // __LINE__
  occa::counterMacro_t counterMacro(&preprocessor); // __COUNTER__

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

  // Missing closing )
  occa::macro_t macro(&preprocessor, "FOO(A) A");
  macro.expand("(1");

  OCCA_TEST_COMPARE(0,
                    !ss.str().size());

  // No macro name
  ss.str("");
  macro.load("");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());

  // Identifier starts badly
  ss.str("");
  macro.load("0A 0");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());

  // No whitespace warning
  ss.str("");
  macro.load("FOO-10");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());

  // Variadic in wrong position
  ss.str("");
  macro.load("FOO(A, ..., B)");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());
}
