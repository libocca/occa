#include "occa/tools/testing.hpp"

#include "preprocessor.hpp"

void testIfElseDefines();
void testErrorDefines();
void testFunctionMacros();
void testEval();

int main(const int argc, const char **argv) {
  testIfElseDefines();
  testErrorDefines();
  testFunctionMacros();
  testEval();
}

void testIfElseDefines() {
  preprocessor_t preprocessor;
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

void testErrorDefines() {
  preprocessor_t preprocessor;
  preprocessor.props["exitOnError"] = false;

  preprocessor.process("#error \"Error\"\n");
  OCCA_TEST_COMPARE(1, preprocessor.errors.size());

  preprocessor.process("#warning \"Warning\"\n");
  OCCA_TEST_COMPARE(1, preprocessor.warnings.size());
}

void testFunctionMacros() {
  preprocessor_t preprocessor;
  preprocessor.props["exitOnError"] = false;

  preprocessor.process("#define FOO(A) A\n");
  OCCA_TEST_COMPARE("",
                    preprocessor.process("FOO\n"));
  OCCA_TEST_COMPARE("",
                    preprocessor.process("FOO()\n"));
  OCCA_TEST_COMPARE("1",
                    preprocessor.process("FOO(1)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A, B) A B\n");
  OCCA_TEST_COMPARE("2 3",
                    preprocessor.process("FOO(2, 3)\n"));
  OCCA_TEST_COMPARE("4 5",
                    preprocessor.process("FOO(4, 5, 6)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A, B) A##B\n");
  OCCA_TEST_COMPARE("6",
                    preprocessor.process("FOO(, 6)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A, ...) A __VA_ARGS__\n");
  OCCA_TEST_COMPARE("7",
                    preprocessor.process("FOO(7,)\n"));
  OCCA_TEST_COMPARE("8 9, 10",
                    preprocessor.process("FOO(8, 9, 10,)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(...) (X, ##__VA_ARGS__)\n");
  OCCA_TEST_COMPARE("(X, 11)",
                    preprocessor.process("FOO(10,)\n"));
  OCCA_TEST_COMPARE("(X)",
                    preprocessor.process("FOO()\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(...) __VA_ARGS_COUNT__\n");
  OCCA_TEST_COMPARE("12",
                    preprocessor.process("FOO (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A) #A\n");
  OCCA_TEST_COMPARE("\"13\"",
                    preprocessor.process("FOO(13)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A, B) #A##B\n");
  OCCA_TEST_COMPARE("\"14\"",
                    preprocessor.process("FOO(1, 4)\n"));
}

void testEval() {
  preprocessor_t preprocessor;
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
