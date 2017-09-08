#include "occa/tools/env.hpp"
#include "occa/tools/testing.hpp"

#include "preprocessor.hpp"

void testMacroDefines();
void testIfElseDefines();
void testErrorDefines();
void testLineDefine();
void testEval();

int main(const int argc, const char **argv) {
  testMacroDefines();
  testErrorDefines();
  testLineDefine();

#if 0
  testIfElseDefines();
  testEval();
#endif
}

void testMacroDefines() {
  occa::preprocessor_t preprocessor;

  preprocessor.process("#define A\n");
  OCCA_TEST_COMPARE("",
                    preprocessor.applyMacros("A"));

  preprocessor.process("#define B 1 2 3\n");
  OCCA_TEST_COMPARE("1 2 3",
                    preprocessor.applyMacros("B"));

  preprocessor.process("#define C(A1) A1\n");
  OCCA_TEST_COMPARE("",
                    preprocessor.applyMacros("C()"));
  OCCA_TEST_COMPARE("1",
                    preprocessor.applyMacros("C(1)"));

  preprocessor.process("#define D(A1, A2) A1 A2\n");
  OCCA_TEST_COMPARE("2 3",
                    preprocessor.applyMacros("D(2, 3)"));
  OCCA_TEST_COMPARE("4 5",
                    preprocessor.applyMacros("D(4, 5, 6)"));

  preprocessor.process("#define E(A1, A2) A1##A2\n");
  OCCA_TEST_COMPARE("6",
                    preprocessor.applyMacros("E(, 6)"));
  OCCA_TEST_COMPARE("07",
                    preprocessor.applyMacros("E(0, 7)"));

  preprocessor.process("#define F(A1, ...) A1 __VA_ARGS__\n");
  OCCA_TEST_COMPARE("7 ",
                    preprocessor.applyMacros("F(7,)"));
  OCCA_TEST_COMPARE("8 9, 10,",
                    preprocessor.applyMacros("F(8, 9, 10,)"));

  preprocessor.process("#define G(...) (X, ##__VA_ARGS__)\n");
  OCCA_TEST_COMPARE("(X, 11 )",
                    preprocessor.applyMacros("G(11,)"));
  OCCA_TEST_COMPARE("(X, )",
                    preprocessor.applyMacros("G()"));

  preprocessor.process("#define H(A1) #A1\n");
  OCCA_TEST_COMPARE("\"12\"",
                    preprocessor.applyMacros("H(12)"));

  preprocessor.process("#define I(A1, A2) #A1##A2\n");
  OCCA_TEST_COMPARE("\"1\"3",
                    preprocessor.applyMacros("I(1, 3)"));
}

#if 0
void testIfElseDefines() {
  occa::preprocessor_t preprocessor;

  OCCA_TEST_COMPARE("",
                    preprocessor.process("#ifdef FOO\n"
                                         "1\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("2",
                    preprocessor.process("#ifndef FOO\n"
                                         "2\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("",
                    preprocessor.process("#if defined(FOO)\n"
                                         "3\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("4",
                    preprocessor.process("#if !defined(FOO)\n"
                                         "4\n"
                                         "#endif\n"));

  preprocessor.process("#define FOO 9\n");

  OCCA_TEST_COMPARE("5",
                    preprocessor.process("#ifdef FOO\n"
                                         "5\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("",
                    preprocessor.process("#ifndef FOO\n"
                                         "6\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("7",
                    preprocessor.process("#if defined(FOO)\n"
                                         "7\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("",
                    preprocessor.process("#if !defined(FOO)\n"
                                         "8\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("9",
                    preprocessor.process("FOO\n"));

  OCCA_TEST_COMPARE("10",
                    preprocessor.process("#undef FOO\n"
                                         "#define FOO 10\n"
                                         "FOO"));

  OCCA_TEST_COMPARE("11",
                    preprocessor.process("#define FOO 11\n"
                                         "FOO"));

  preprocessor.process("#undef FOO\n");

  OCCA_TEST_COMPARE("",
                    preprocessor.process("#ifdef FOO\n"
                                         "12\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("13",
                    preprocessor.process("#ifndef FOO\n"
                                         "13\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("",
                    preprocessor.process("#if defined(FOO)\n"
                                         "14\n"
                                         "#endif\n"));

  OCCA_TEST_COMPARE("15",
                    preprocessor.process("#if !defined(FOO)\n"
                                         "15\n"
                                         "#endif\n"));
}
#endif

void testErrorDefines() {
  occa::preprocessor_t preprocessor;
  std::stringstream ss;
  preprocessor.exitOnFatalError = false;
  preprocessor.setOutputStream(ss);

  const bool printOutput = true;

  preprocessor.processSource("#error \"Error\"\n");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  preprocessor.clear();
  preprocessor.processSource("#warning \"Warning\"\n");
  OCCA_TEST_COMPARE(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");
}

void testLineDefine() {
  occa::preprocessor_t preprocessor;

  preprocessor.process("#line 10\n");
  OCCA_TEST_COMPARE(10,
                    preprocessor.currentFrame.lineNumber);

  preprocessor.process("#line 20 \"foo\"\n");
  OCCA_TEST_COMPARE(20,
                    preprocessor.currentFrame.lineNumber);
  OCCA_TEST_COMPARE(occa::env::PWD + "foo",
                    preprocessor.currentFrame.filename());
}

#if 0
void testEval() {
  occa::preprocessor_t preprocessor;
  preprocessor.props["exitOnError"] = false;

  // Types
  OCCA_TEST_COMPARE<int>(1 + 1,
                         preprocessor.eval<int>("1 + 1"));
  OCCA_TEST_COMPARE<bool>(true,
                          preprocessor.eval<bool>("1 + 1"));
  OCCA_TEST_COMPARE<double>(0.5 + 1.5,
                            preprocessor.eval<double>("0.5 + 1.5"));
  OCCA_TEST_COMPARE("2",
                    preprocessor.eval<std::string>("1 + 1"));
  OCCA_TEST_COMPARE<double>(100000000000L,
                            preprocessor.eval<long>("100000000000L"));

  // Unary Operators
  OCCA_TEST_COMPARE<int>(+1,
                         preprocessor.eval<int>("+1"));
  OCCA_TEST_COMPARE<int>(-1,
                         preprocessor.eval<int>("-1"));
  OCCA_TEST_COMPARE<int>(!1,
                         preprocessor.eval<int>("!1"));
  OCCA_TEST_COMPARE<int>(~1,
                         preprocessor.eval<int>("~1"));

  // Binary Operators
  OCCA_TEST_COMPARE<double>(1 + 2,
                            preprocessor.eval<double>("1 + 2"));
  OCCA_TEST_COMPARE<double>(1 - 2,
                            preprocessor.eval<double>("1 - 2"));
  OCCA_TEST_COMPARE<int>(1 / 2,
                         preprocessor.eval<double>("1 / 2"));
  OCCA_TEST_COMPARE<double>(1 / 2.0,
                            preprocessor.eval<double>("1 / 2.0"));
  OCCA_TEST_COMPARE<int>(5 % 2,
                         preprocessor.eval<int>("5 % 2"));

  // Bit operators
  OCCA_TEST_COMPARE<int>(1 & 3,
                         preprocessor.eval<int>("1 & 3"));
  OCCA_TEST_COMPARE<int>(1 ^ 3,
                         preprocessor.eval<int>("1 ^ 3"));
  OCCA_TEST_COMPARE<int>(1 | 3,
                         preprocessor.eval<int>("1 | 3"));

  // Shift operators
  OCCA_TEST_COMPARE<int>(0,
                         preprocessor.eval<int>("1 >> 1"));
  OCCA_TEST_COMPARE<int>(2,
                         preprocessor.eval<int>("1 << 1"));

  // Parentheses
  OCCA_TEST_COMPARE<double>(3*(1 + 1),
                            preprocessor.eval<int>("3*(1 + 1)"));

  OCCA_TEST_COMPARE<double>((3 * 1)*(1 + 1),
                            preprocessor.eval<int>("(3 * 1)*(1 + 1)"));

  /*
  // NaN and Inf
  float x = preprocessor.eval<float>("1/0");
  OCCA_TEST_COMPARE(true, x != x);

  x = preprocessor.eval<float>("0/0");
  OCCA_TEST_COMPARE(true, x != x);
  */
}
#endif
