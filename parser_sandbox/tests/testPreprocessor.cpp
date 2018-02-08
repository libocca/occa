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
#include "occa/tools/env.hpp"

#include "./tokenUtils.hpp"

void setStream(const std::string &s) {
  tu::setStream(s);
  stream = (stream
            .map(new occa::lang::preprocessor())
            .map(new occa::lang::mergeStrings()));
}

void setToken(const std::string &s) {
  setStream(s);
  tu::getToken();
}

class preprocessorTester {
  occa::lang::preprocessor pp;

public:
  preprocessorTester();

  void testMacroDefines();
  void testIfElseDefines();
  void testErrorDefines();
  void testWeirdCase();
  void testSpecialMacros();
  void testEval();

  void test() {
    testMacroDefines();
    testErrorDefines();
    testSpecialMacros();
    testWeirdCase();
    testIfElseDefines();
    testEval();
  }
};

int main(const int argc, const char **argv) {
  preprocessorTester tester;
  tester.test();
}

preprocessorTester::preprocessorTester() {}

void preprocessorTester::testMacroDefines() {
  #if 0
  OCCA_ASSERT_EQUAL("",
                    pp.processSource("#define A\n"
                                     "A"));

  OCCA_ASSERT_EQUAL("1 2 3",
                    pp.processSource("#define B 1 2 3\n"
                                     "B"));

  pp.processSource("#define C(A1) A1\n");
  OCCA_ASSERT_EQUAL("",
                    pp.applyMacros("C()"));
  OCCA_ASSERT_EQUAL("1",
                    pp.applyMacros("C(1)"));

  pp.processSource("#define D(A1, A2) A1 A2\n");
  OCCA_ASSERT_EQUAL("2 3",
                    pp.applyMacros("D(2, 3)"));
  OCCA_ASSERT_EQUAL("4 5",
                    pp.applyMacros("D(4, 5, 6)"));

  pp.processSource("#define E(A1, A2) A1##A2\n");
  OCCA_ASSERT_EQUAL("6",
                    pp.applyMacros("E(, 6)"));
  OCCA_ASSERT_EQUAL("07",
                    pp.applyMacros("E(0, 7)"));

  pp.processSource("#define F(A1, ...) A1 __VA_ARGS__\n");
  OCCA_ASSERT_EQUAL("7 ",
                    pp.applyMacros("F(7,)"));
  OCCA_ASSERT_EQUAL("8 9, 10,",
                    pp.applyMacros("F(8, 9, 10,)"));

  pp.processSource("#define G(...) (X, ##__VA_ARGS__)\n");
  OCCA_ASSERT_EQUAL("(X, 11 )",
                    pp.applyMacros("G(11,)"));
  OCCA_ASSERT_EQUAL("(X, )",
                    pp.applyMacros("G()"));

  pp.processSource("#define H(A1) #A1\n");
  OCCA_ASSERT_EQUAL("\"12\"",
                    pp.applyMacros("H(12)"));

  pp.processSource("#define I(A1, A2) #A1##A2\n");
  OCCA_ASSERT_EQUAL("\"1\"3",
                    pp.applyMacros("I(1, 3)"));
#endif
}

void preprocessorTester::testIfElseDefines() {
  #if 0
  OCCA_ASSERT_EQUAL("",
                    pp.processSource("#ifdef FOO\n"
                                     "1\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("2",
                    pp.processSource("#ifndef FOO\n"
                                     "2\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("",
                    pp.processSource("#if defined(FOO)\n"
                                     "3\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("4",
                    pp.processSource("#if !defined(FOO)\n"
                                     "4\n"
                                     "#endif\n"));

  pp.processSource("#define FOO 9\n");

  OCCA_ASSERT_EQUAL("5",
                    pp.processSource("#ifdef FOO\n"
                                     "5\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("",
                    pp.processSource("#ifndef FOO\n"
                                     "6\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("7",
                    pp.processSource("#if defined(FOO)\n"
                                     "7\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("",
                    pp.processSource("#if !defined(FOO)\n"
                                     "8\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("9",
                    pp.processSource("FOO\n"));

  OCCA_ASSERT_EQUAL("10",
                    pp.processSource("#undef FOO\n"
                                     "#define FOO 10\n"
                                     "FOO"));

  OCCA_ASSERT_EQUAL("11",
                    pp.processSource("#define FOO 11\n"
                                     "FOO"));

  pp.processSource("#undef FOO\n");

  OCCA_ASSERT_EQUAL("",
                    pp.processSource("#ifdef FOO\n"
                                     "12\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("13",
                    pp.processSource("#ifndef FOO\n"
                                     "13\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("",
                    pp.processSource("#if defined(FOO)\n"
                                     "14\n"
                                     "#endif\n"));

  OCCA_ASSERT_EQUAL("15",
                    pp.processSource("#if !defined(FOO)\n"
                                     "15\n"
                                     "#endif\n"));
#endif
}

void preprocessorTester::testWeirdCase() {
  #if 0
  // Should print out "x ## y"
  // std::string str = pp.processSource("#define hash_hash # ## #\n"
  //                                              "#define mkstr(a) # a\n"
  //                                              "#define in_between(a) mkstr(a)\n"
  //                                              "#define join(c, d) in_between(c hash_hash d)\n"
  //                                              "join(x, y)\n");
  // std::cout << "str = " << str << '\n';
#endif
}

void preprocessorTester::testErrorDefines() {
  #if 0
  std::stringstream ss;
  pp.exitOnFatalError = false;
  pp.setOutputStream(ss);

  const bool printOutput = true;

  pp.processSource("#error \"Error\"\n");
  OCCA_ASSERT_EQUAL(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");

  pp.clear();
  pp.exitOnFatalError = false;
  pp.setOutputStream(ss);

  pp.processSource("#warning \"Warning\"\n");
  OCCA_ASSERT_EQUAL(0,
                    !ss.str().size());
  if (printOutput) {
    std::cout << ss.str();
  }
  ss.str("");
#endif
}

void preprocessorTester::testSpecialMacros() {
  #if 0
  OCCA_ASSERT_EQUAL("10\n",
                    pp.processSource("#line 10\n"
                                     "__LINE__\n"));

  OCCA_ASSERT_EQUAL("20\n"
                    + occa::env::PWD + "foo\n"
                    "22\n",
                    pp.processSource("#line 20 \"foo\"\n"
                                     "__LINE__\n"
                                     "__FILE__\n"
                                     "__LINE__\n"));

  OCCA_ASSERT_EQUAL("0\n"
                    "1\n"
                    "3\n"
                    "2\n",
                    pp.processSource("__COUNTER__\n"
                                     "__COUNTER__\n"
                                     "__LINE__\n"
                                     "__COUNTER__\n"));

  std::cout << pp.processSource("_DATE_ macro: __DATE__\n"
                                "_TIME_ macro: __TIME__\n");
#endif
}

void preprocessorTester::testEval() {
  #if 0
  // Types
  OCCA_ASSERT_EQUAL<int>(1 + 1,
                         pp.eval<int>("1 + 1"));
  OCCA_ASSERT_EQUAL<bool>(true,
                          pp.eval<bool>("1 + 1"));
  OCCA_ASSERT_EQUAL<double>(0.5 + 1.5,
                            pp.eval<double>("0.5 + 1.5"));
  OCCA_ASSERT_EQUAL("2",
                    pp.eval<std::string>("1 + 1"));
  OCCA_ASSERT_EQUAL<double>(100000000000L,
                            pp.eval<long>("100000000000L"));

  // Unary Operators
  OCCA_ASSERT_EQUAL<int>(+1,
                         pp.eval<int>("+1"));
  OCCA_ASSERT_EQUAL<int>(-1,
                         pp.eval<int>("-1"));
  OCCA_ASSERT_EQUAL<int>(!1,
                         pp.eval<int>("!1"));
  OCCA_ASSERT_EQUAL<int>(~1,
                         pp.eval<int>("~1"));

  // Binary Operators
  OCCA_ASSERT_EQUAL<double>(1 + 2,
                            pp.eval<double>("1 + 2"));
  OCCA_ASSERT_EQUAL<double>(1 - 2,
                            pp.eval<double>("1 - 2"));
  OCCA_ASSERT_EQUAL<int>(1 / 2,
                         pp.eval<double>("1 / 2"));
  OCCA_ASSERT_EQUAL<double>(1 / 2.0,
                            pp.eval<double>("1 / 2.0"));
  OCCA_ASSERT_EQUAL<int>(5 % 2,
                         pp.eval<int>("5 % 2"));

  // Bit operators
  OCCA_ASSERT_EQUAL<int>(1 & 3,
                         pp.eval<int>("1 & 3"));
  OCCA_ASSERT_EQUAL<int>(1 ^ 3,
                         pp.eval<int>("1 ^ 3"));
  OCCA_ASSERT_EQUAL<int>(1 | 3,
                         pp.eval<int>("1 | 3"));

  // Shift operators
  OCCA_ASSERT_EQUAL<int>(0,
                         pp.eval<int>("1 >> 1"));
  OCCA_ASSERT_EQUAL<int>(2,
                         pp.eval<int>("1 << 1"));

  // Parentheses
  OCCA_ASSERT_EQUAL<double>(3*(1 + 1),
                            pp.eval<int>("3*(1 + 1)"));

  OCCA_ASSERT_EQUAL<double>((3 * 1)*(1 + 1),
                            pp.eval<int>("(3 * 1)*(1 + 1)"));

  /*
  // NaN and Inf
  float x = pp.eval<float>("1/0");
  OCCA_ASSERT_EQUAL(true, x != x);

  x = pp.eval<float>("0/0");
  OCCA_ASSERT_EQUAL(true, x != x);
  */
#endif
}
