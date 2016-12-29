#include "/Users/dsm5/git/night/examples/foo/tests/test.cpp"
#include "/Users/dsm5/git/night/examples/foo/preprocessor.hpp"

void testIfElseDefines() {
  preprocessor_t preprocessor;
  test::compare("",
                preprocessor.process("#ifdef FOO\n"
                                     "1\n"
                                     "#endif\n"));

  test::compare("2",
                preprocessor.process("#ifndef FOO\n"
                                     "2\n"
                                     "#endif\n"));

  test::compare("",
                preprocessor.process("#if defined(FOO)\n"
                                     "3\n"
                                     "#endif\n"));

  test::compare("4",
                preprocessor.process("#if !defined(FOO)\n"
                                     "4\n"
                                     "#endif\n"));

  preprocessor.process("#define FOO 9\n");

  test::compare("5",
                preprocessor.process("#ifdef FOO\n"
                                     "5\n"
                                     "#endif\n"));

  test::compare("",
                preprocessor.process("#ifndef FOO\n"
                                     "6\n"
                                     "#endif\n"));

  test::compare("7",
                preprocessor.process("#if defined(FOO)\n"
                                     "7\n"
                                     "#endif\n"));

  test::compare("",
                preprocessor.process("#if !defined(FOO)\n"
                                     "8\n"
                                     "#endif\n"));

  test::compare("9",
                preprocessor.process("FOO\n"));

  test::compare("10",
                preprocessor.process("#undef FOO\n"
                                     "#define FOO 10\n"
                                     "FOO"));

  test::compare("11",
                preprocessor.process("#define FOO 11\n"
                                     "FOO"));

  preprocessor.process("#undef FOO\n");

  test::compare("",
                preprocessor.process("#ifdef FOO\n"
                                     "12\n"
                                     "#endif\n"));

  test::compare("13",
                preprocessor.process("#ifndef FOO\n"
                                     "13\n"
                                     "#endif\n"));

  test::compare("",
                preprocessor.process("#if defined(FOO)\n"
                                     "14\n"
                                     "#endif\n"));

  test::compare("15",
                preprocessor.process("#if !defined(FOO)\n"
                                     "15\n"
                                     "#endif\n"));
}

void testErrorDefines() {
  preprocessor_t preprocessor;
  preprocessor.props["exitOnError"] = false;

  preprocessor.process("#error \"Error\"\n");
  test::compare(1, preprocessor.errors.size());

  preprocessor.process("#warning \"Warning\"\n");
  test::compare(1, preprocessor.warnings.size());
}

void testFunctionMacros() {
  preprocessor_t preprocessor;
  preprocessor.props["exitOnError"] = false;

  preprocessor.process("#define FOO(A) A\n");
  test::compare("",
                preprocessor.process("FOO\n"));
  test::compare("",
                preprocessor.process("FOO()\n"));
  test::compare("1",
                preprocessor.process("FOO(1)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A, B) A B\n");
  test::compare("2 3",
                preprocessor.process("FOO(2, 3)\n"));
  test::compare("4 5",
                preprocessor.process("FOO(4, 5, 6)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A, B) A##B\n");
  test::compare("6",
                preprocessor.process("FOO(, 6)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A, ...) A __VA_ARGS__\n");
  test::compare("7",
                preprocessor.process("FOO(7,)\n"));
  test::compare("8 9, 10",
                preprocessor.process("FOO(8, 9, 10,)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(...) (X, ##__VA_ARGS__)\n");
  test::compare("(X, 11)",
                preprocessor.process("FOO(10,)\n"));
  test::compare("(X)",
                preprocessor.process("FOO()\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(...) __VA_ARGS_COUNT__\n");
  test::compare("12",
                preprocessor.process("FOO (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A) #A\n");
  test::compare("\"13\"",
                preprocessor.process("FOO(13)\n"));

  preprocessor.process("#undef FOO\n"
                       "#define FOO(A, B) #A##B\n");
  test::compare("\"14\"",
                preprocessor.process("FOO(1, 4)\n"));
}

void testEval() {
  preprocessor_t preprocessor;
  preprocessor.props["exitOnError"] = false;

  // Types
  test::compare<int>(1 + 1,
                     preprocessor.eval<int>("1 + 1"));
  test::compare<bool>(true,
                      preprocessor.eval<bool>("1 + 1"));
  test::compare<double>(0.5 + 1.5,
                        preprocessor.eval<double>("0.5 + 1.5"));
  test::compare("2",
                preprocessor.eval<std::string>("1 + 1"));
  test::compare<double>(100000000000L,
                        preprocessor.eval<long>("100000000000L"));

  // Unary Operators
  test::compare<int>(+1,
                     preprocessor.eval<int>("+1"));
  test::compare<int>(-1,
                     preprocessor.eval<int>("-1"));
  test::compare<int>(!1,
                     preprocessor.eval<int>("!1"));
  test::compare<int>(~1,
                     preprocessor.eval<int>("~1"));

  // Binary Operators
  test::compare<double>(1 + 2,
                        preprocessor.eval<double>("1 + 2"));
  test::compare<double>(1 - 2,
                        preprocessor.eval<double>("1 - 2"));
  test::compare<int>(1 / 2,
                     preprocessor.eval<double>("1 / 2"));
  test::compare<double>(1 / 2.0,
                        preprocessor.eval<double>("1 / 2.0"));
  test::compare<int>(5 % 2,
                     preprocessor.eval<int>("5 % 2"));

  // Bit operators
  test::compare<int>(1 & 3,
                     preprocessor.eval<int>("1 & 3"));
  test::compare<int>(1 ^ 3,
                     preprocessor.eval<int>("1 ^ 3"));
  test::compare<int>(1 | 3,
                     preprocessor.eval<int>("1 | 3"));

  // Shift operators
  test::compare<int>(0,
                     preprocessor.eval<int>("1 >> 1"));
  test::compare<int>(2,
                     preprocessor.eval<int>("1 << 1"));

  // Parentheses
  test::compare<double>(3*(1 + 1),
                        preprocessor.eval<int>("3*(1 + 1)"));

  test::compare<double>((3 * 1)*(1 + 1),
                        preprocessor.eval<int>("(3 * 1)*(1 + 1)"));

  /*
  // NaN and Inf
  float x = preprocessor.eval<float>("1/0");
  test::compare(true, x != x);

  x = preprocessor.eval<float>("0/0");
  test::compare(true, x != x);
  */
}