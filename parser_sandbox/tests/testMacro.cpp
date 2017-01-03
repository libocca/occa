#include <sstream>

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
}

void testPlainMacros() {
  occa::preprocessor_t preprocessor;
  occa::macro_t macro(&preprocessor, "A");

  OCCA_TEST_COMPARE(macro.name, "A");
  OCCA_TEST_COMPARE(macro.expand(""), "");

  macro.load("B 1 2 3");
  OCCA_TEST_COMPARE(macro.name, "B");
  OCCA_TEST_COMPARE(macro.expand(""), "1 2 3");

  macro.load("");
  OCCA_TEST_COMPARE(macro.name, "");
  OCCA_TEST_COMPARE(macro.expand(""), "");
}

void testFunctionMacros() {
  occa::preprocessor_t preprocessor;
  occa::macro_t macro(&preprocessor, "FOO(A) A");

  OCCA_TEST_COMPARE("",
                    macro.expand("()"));
  OCCA_TEST_COMPARE("1",
                    macro.expand("(1)"));

  macro.load("FOO(A, B) A B");
  OCCA_TEST_COMPARE("2  3",
                    macro.expand("(2, 3)"));
  OCCA_TEST_COMPARE("4  5",
                    macro.expand("(4, 5, 6)"));

  macro.load("FOO(A, B) A##B");
  OCCA_TEST_COMPARE(" 6",
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
  preprocessor.filename   = "foo";
  preprocessor.lineNumber = 10;

  char *c = new char[1];

  occa::fileMacro_t fileMacro(&preprocessor);       // __FILE__
  occa::lineMacro_t lineMacro(&preprocessor);       // __LINE__
  occa::counterMacro_t counterMacro(&preprocessor); // __COUNTER__

  OCCA_TEST_COMPARE(occa::toString(preprocessor.filename),
                    fileMacro.expand(c));

  OCCA_TEST_COMPARE(occa::toString(preprocessor.lineNumber),
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
  preprocessor.setOutputStream(ss);

  occa::macro_t macro(&preprocessor, "FOO(A) A");
  macro.expand("(1");

  OCCA_TEST_COMPARE(0,
                    !ss.str().size());
}
