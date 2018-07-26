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

#include <occa/tools/env.hpp>
#include <occa/tools/testing.hpp>

#include <occa/lang/tokenizer.hpp>
#include <occa/lang/processingStages.hpp>
#include <occa/lang/preprocessor.hpp>

void testMacroDefines();
void testCppStandardTests();
void testIfElse();
void testIfElseDefines();
void testErrorDefines();
void testSpecialMacros();
void testInclude();
void testPragma();
void testOccaPragma();

using namespace occa::lang;

//---[ Util Methods ]-------------------
std::string source;
tokenizer_t tokenizer;
preprocessor_t preprocessor;
newlineTokenFilter newlineFilter;
occa::lang::stream<token_t*> tokenStream = (
  tokenizer
  .map(preprocessor)
  .map(newlineFilter)
);
token_t *token;

void setStream(const std::string &s) {
  delete token;
  token = NULL;
  ::source = s;
  tokenizer.set(source.c_str());
  preprocessor.clear();
}

void getToken() {
  delete token;
  token = NULL;
  tokenStream >> token;
}

void setToken(const std::string &s) {
  setStream(s);
  getToken();
}

int getTokenType() {
  return token ? token->type() : 0;
}

occa::primitive nextTokenPrimitiveValue() {
  getToken();
  return token->to<primitiveToken>().value;
}

std::string nextTokenStringValue() {
  getToken();
  while (token &&
         (token->type() == tokenType::newline)) {
    getToken();
  }
  return token->to<stringToken>().value;
}
//======================================

//---[ Tests ]--------------------------
int main(const int argc, const char **argv) {
  testMacroDefines();
  testCppStandardTests();
  testIfElse();
  testIfElseDefines();
  testErrorDefines();
  testSpecialMacros();
  testInclude();
  testPragma();
  testOccaPragma();

  delete token;
}

void testMacroDefines() {
  std::cerr << "Testing preprocessor errors:\n";
  setStream(
    // Test #define
    "#define A\n"
    "A\n"
    // Test multi-token expansion
    "#define B 1 2 3\n"
    "B\n"
    // Test function-like macros
    "#define C(A1) A1\n"
    "C(1)\n"
    // Test multi-argument expansion
    "#define D(A1, A2) A1 A2 A2 A1\n"
    "D(2, 3)\n"
    // Test stringify
    "#define E(A1) #A1\n"
    "E(1    2 3     ....   /path/to/somewhere  )\n"
    "E(# ## ### #### ### ## #)\n"
    // Test multi-token stringify
    "#define F(A1, A2) #A1 #A2\n"
    "F(12, 34)\n"
    // Test concat
    "#define G(A1, A2) A1 ## A2\n"
    "G(, 6)\n"
    "G(0, 7)\n"
    // Test varargs
    "#define H(A1, ...) A1 __VA_ARGS__\n"
    "H(7,)\n"
    "H(8, 9, 10,)\n"
    // Test only varargs
    "#define I(...) 5 ##__VA_ARGS__\n"
    "I(11,)\n"
    "I()\n"
    // Errors:
    // - Argument missing
    "C()\n"
    // - Too many arguments
    "D(4, 5, 6)\n"
    // - Test stringify with concat fail
    "#define J(A1, A2) # A1 ## A2\n"
    "J(1, 3)\n"
  );
  while (!tokenStream.isEmpty()) {
    getToken();
  }

  // Test error counting
  preprocessor_t &pp = *((preprocessor_t*) tokenStream.getInput("preprocessor_t"));
  ASSERT_EQ(3,
            pp.errors);

  // Make sure we can handle recursive macros
#define identifier (token->to<identifierToken>().value)
  setStream(
    "#define foo foo\n"
    "foo"
  );
  getToken();
  setStream(
    "#define foo3 foo2\n"
    "#define foo2 foo1\n"
    "#define foo1 foo2\n"
    "foo1"
    " foo2"
    " foo3"
  );
  getToken();
  ASSERT_EQ("foo1",
            identifier);
  getToken();
  ASSERT_EQ("foo2",
            identifier);
  getToken();
  ASSERT_EQ("foo2",
            identifier);

#undef identifier
}

void testCppStandardTests() {
  // Test 1 in the C++ standard
  setStream(
    "#define hash_hash # ## #\n"
    "#define mkstr(a) # a\n"
    "#define in_between(a) mkstr(a)\n"
    "#define join(c, d) in_between(c hash_hash d)\n"
    "join(x, y)"
  );

  getToken();
  const std::string output = token->to<stringToken>().value;
  ASSERT_EQ("x ## y",
            output);

  // Test 2 in C++ standard
  setStream(
    // Defines
    "#define x      3\n"
    "#define f(a)   f(x * (a))\n"
    "#udnef  x\n"
    "#define x      2\n"
    "#define g      f\n"
    "#define z      z[0]\n"
    "#define h      g(~\n"
    "#define m(a)   a(w)\n"
    "#define w      0,1\n"
    "#define t(a)   a\n"
    "#define p()    int\n"
    "#define q(x)   x\n"
    "#define r(x,y) x ## y\n"
    "#define str(x) # x\n"
    // Source
    "f(y+1) + f(f(z)) % t(t(g)(0) + t)(1);\n"
    "g(x+(3,4)-w) | h 5) & m\n"
    "    (f)^m(m);\n"
    "p() i[q()] = { q(1), r(2,3), r(4,), r(,5), r(,) };\n"
    "char c[2][6] = { str(hello), str() };\n"
  );

  /*
    f(2 * y+1)) + f(2 * (f(2 * (z[0])))) % f(2 * (0)) + t(1);
    f(2 * (2+(3,4)-0,1)) | f(2 * (~ 5)) & f(2 * (0,1))^m(0,1);
    int i[] = { 1, 23, 4, 5, };
    char c[2][6] = { "hello", "" };
  */

  // Test 3 in C++ standard
  setStream(
    // Defines
    "#define str(s)      \n"
    "#define xstr(x)     \n"
    "#define debug(s, t) \n"
    "#define INCFILE(n)  \n"
    "#define glue(a, b)  \n"
    "#define xglue(a, b) \n"
    "#define HIGHLOW     \n"
    "#define LOW         \n"
    // Source
    "debug(1, 2);\n"
    "fputs(str(strncmp(\"abc\0d\", \"abc\", '\4')\n"
    "    == 0) str(: @\n), s);\n"
    "include xstr(INCFILE(2).h)\n"
    "glue(HIGH, LOW);\n"
    "xglue(HIGH, LOW)\n"
  );

  /*
    printf("x1= %d, x2= %s", x1, x2);
    fputs("strncmp(\"abc\\0d\", \"abc\", '\\4') == 0: @\n", s);
    include "vers2.h"
    "hello";
    "hello, world"
  */

  // Test 4 in C++ standard
  setStream(
    // Defines
    "#define t(x,y,z) x ## y ## z\n"
    // Source
    "int j[] = { t(1,2,3), t(,4,5), t(6,,7), t(8,9,),\n"
    "t(10,,), t(,11,), t(,,12), t(,,) };\n"
  );

  /*
    int j[] = { 123, 45, 67, 89,
    10, 11, 12, };
  */

  // Test 5 in C++ standard
  setStream(
    // Defines
    "#define OBJ_LIKE      (1-1)\n"
    "#define OBJ_LIKE       /* white space */ (1-1) /* other */\n"
    "#define FUNC_LIKE(a)    ( a )\n"
    "#define FUNC_LIKE( a )(      /* note the white space */ \\\n"
    "                a /* other stuff on this line\n"
    "                  */ )\n"
  );

  std::cerr << "Testing wrong macro redefinitions:\n";
  setStream(
    // Defines
    "#define OBJ_LIKE      (1-1)\n"
    "#define OBJ_LIKE      (0)\n"
    "#define FUNC_LIKE(a)  ( a )\n"
    "#define FUNC_LIKE(b)  ( a )\n"
  );

  // Test 6 in C++ standard
  setStream(
    // Defines
    "#define debug(...) fprintf(stderr, __VA_ARGS__)\n"
    "#define showlist(...) puts(#__VA_ARGS__)\n"
    "#define report(test, ...) ((test) ? puts(#test) : printf(__VA_ARGS__))\n"
    // Source
    "debug(\"Flag\");\n"
    "debug(\"X = %d\\n\", x);\n"
    "showlist(The first, second, and third items.);\n"
    "report(x>y, \"x is %d but y is %d\", x, y);\n"
  );

  /*
    fprintf(stderr, "Flag");
    fprintf(stderr, "X = %d\n", x);
    puts("The first, second, and third items.");
    ((x>y) ? puts("x>y") : printf("x is %d but y is %d", x, y));
  */
}

void testIfElse() {
  setStream(
    // Test #if true with #elif false
    "#if true\n"
    "1\n"
    "#elif false\n"
    "A\n"
    "#elif false\n"
    "B\n"
    "#else\n"
    "C\n"
    "#endif\n"
    // Test #if true with #elif true
    "#if true\n"
    "1\n"
    "#elif true\n"
    "D\n"
    "#elif true\n"
    "E\n"
    "#else\n"
    "F\n"
    "#endif\n"
    // Test #if false with 1st #elif true
    "#if false\n"
    "G\n"
    "#elif true\n"
    "1\n"
    "#elif false\n"
    "H\n"
    "#else\n"
    "I\n"
    "#endif\n"
    // Test #if false with 2nd #elif true
    "#if false\n"
    "J\n"
    "#elif false\n"
    "K\n"
    "#elif true\n"
    "1\n"
    "#else\n"
    "L\n"
    "#endif\n"
    // Test #else without #elif
    "#if false\n"
    "M\n"
    "#else\n"
    "1\n"
    "#endif\n"
    // Test #else without #elif
    "#if false\n"
    "N\n"
    "#elif false\n"
    "O\n"
    "#elif false\n"
    "P\n"
    "#else\n"
    "1\n"
    "#endif\n");

  int tokensFound = 0;
  do {
    getToken();
    if (getTokenType() & tokenType::newline) {
      continue;
    }
    if (token) {
      ++tokensFound;
      ASSERT_EQ_BINARY(tokenType::primitive,
                       token->type());
      ASSERT_EQ(1,
                (int) token->to<primitiveToken>().value);
    }
  } while (token);

  ASSERT_EQ(6,
            tokensFound);
}

void testIfElseDefines () {
  // Test defines
  setStream(
    // ""
    "#ifdef FOO\n"
    "1\n"
    "#endif\n"
    // "2"
    "#ifndef FOO\n"
    "2\n"
    "#endif\n"
    // ""
    "#if defined(FOO)\n"
    "3\n"
    "#endif\n"
    // "4"
    "#if !defined(FOO)\n"
    "4\n"
    "#endif\n"
    // Redefine FOO
    "#define FOO 9\n"
    // "5"
    "#ifdef FOO\n"
    "5\n"
    "#endif\n"
    // ""
    "#ifndef FOO\n"
    "6\n"
    "#endif\n"
    // "7"
    "#if defined(FOO)\n"
    "7\n"
    "#endif\n"
    // ""
    "#if !defined(FOO)\n"
    "8\n"
    "#endif\n"
    // "9"
    "FOO\n"
    // "10"
    "#undef FOO\n"
    "#define FOO 10\n"
    "FOO\n"
    // "11"
    "#define FOO 11\n"
    "FOO\n"
    "#undef FOO\n"
    // ""
    "#ifdef FOO\n"
    "12\n"
    "#endif\n"
    // "13"
    "#ifndef FOO\n"
    "13\n"
    "#endif\n"
    // ""
    "#if defined(FOO)\n"
    "14\n"
    "#endif\n"
    // "15"
    "#if !defined(FOO)\n"
    "15\n"
    "#endif\n"
  );
  int values[9] = {
    2, 4, 5, 7, 9, 10, 11, 13, 15
  };
  for (int i = 0; i < 9; ++i) {
    while (true) {
      getToken();
      if (getTokenType() & tokenType::primitive) {
        break;
      }
      if (!getTokenType()) {
        OCCA_FORCE_ERROR("[" << i << "] Expected more tokens");
      }
      if (getTokenType() != tokenType::newline) {
        token->printError("Expected only primitive or newline tokens");
        OCCA_FORCE_ERROR("Error on [" << i << "]");
      }
    }
    primitiveToken &pToken = *((primitiveToken*) token);
    ASSERT_EQ(values[i],
              (int) pToken.value);
  }
  while (!tokenStream.isEmpty()) {
    getToken();
  }
}

void testErrorDefines() {
  std::cerr << "Testing error and warning directives\n";
  setStream(
    "#error \"Testing #error\"\n"
    "#warning \"Testing #warning\"\n"
  );
  while (!tokenStream.isEmpty()) {
    getToken();
  }
}

void testSpecialMacros() {
  setStream(
    "__COUNTER__\n"
    "__FILE__ __LINE__\n"
    "__COUNTER__\n"
    "__FILE__ __LINE__ __LINE__\n"
    "#line 20\n"
    "__LINE__\n"
    "#line 30 \"foobar\"\n"
    "__FILE__ __LINE__\n"
    "__COUNTER__\n"
    "__DATE__ __TIME__\n"
  );

  // __COUNTER__
  ASSERT_EQ(0,
            (int) nextTokenPrimitiveValue());

  // __FILE__ __LINE__
  ASSERT_EQ("(source)",
            nextTokenStringValue());
  ASSERT_EQ(2,
            (int) nextTokenPrimitiveValue());

  // __COUNTER__
  ASSERT_EQ(1,
            (int) nextTokenPrimitiveValue());

  // __FILE__ __LINE__ __LINE__
  ASSERT_EQ("(source)",
            nextTokenStringValue());
  ASSERT_EQ(4,
            (int) nextTokenPrimitiveValue());
  ASSERT_EQ(4,
            (int) nextTokenPrimitiveValue());

  // __LINE__
  ASSERT_EQ(20,
            (int) nextTokenPrimitiveValue());

  // __FILE__ __LINE__
  ASSERT_EQ("foobar",
            nextTokenStringValue());
  ASSERT_EQ(30,
            (int) nextTokenPrimitiveValue());

  // __COUNTER__
  ASSERT_EQ(2,
            (int) nextTokenPrimitiveValue());

  // __DATE__ __TIME__
  getToken();
  ASSERT_EQ_BINARY(tokenType::string,
                   token->type());
  getToken();
  ASSERT_EQ_BINARY(tokenType::string,
                   token->type());

  while(!tokenStream.isEmpty()) {
    getToken();
  }
}

void testInclude() {
  const std::string testFile = (occa::env::OCCA_DIR
                                + "tests/files/preprocessor.hpp");

  std::stringstream ss;
  ss << "#include \"" << testFile << "\"\n"
     << "#include <"  << testFile << ">\n"
     << "#define ERROR true\n"
     << "#include \"" << testFile << "\"\n";
  setStream(ss.str());

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(j,
                (int) nextTokenPrimitiveValue());
    }
  }
  // Error out in the last include
  while(!tokenStream.isEmpty()) {
    getToken();
  }

  preprocessor_t &pp = *((preprocessor_t*) tokenStream.getInput("preprocessor_t"));
  ASSERT_EQ(1,
            (int) pp.dependencies.size());
}

void testPragma() {
  setStream("#pragma\n");
  getToken();
  ASSERT_EQ_BINARY(tokenType::pragma,
                   token->type());
  ASSERT_EQ(0,
            (int) token->to<pragmaToken>().value.size());

  setStream("#pragma");
  getToken();
  ASSERT_EQ_BINARY(tokenType::pragma,
                   token->type());
  ASSERT_EQ(0,
            (int) token->to<pragmaToken>().value.size());

  setStream("#pragma foo\n");
  getToken();
  ASSERT_EQ_BINARY(tokenType::pragma,
                   token->type());
  ASSERT_EQ("foo",
            token->to<pragmaToken>().value);

  setStream("#pragma foo 1 2 3\n");
  getToken();
  ASSERT_EQ_BINARY(tokenType::pragma,
                   token->type());

  ASSERT_EQ("foo 1 2 3",
            token->to<pragmaToken>().value);

  setStream("#pragma foo 1 2 3");
  getToken();
  ASSERT_EQ_BINARY(tokenType::pragma,
                   token->type());

  ASSERT_EQ("foo 1 2 3",
            token->to<pragmaToken>().value);
}

void testOccaPragma() {
#define checkOp(op_)                            \
  getToken();                                   \
  ASSERT_EQ_BINARY(tokenType::op,               \
                   token->type());              \
  ASSERT_EQ(op_,                                \
            token->getOpType())

#define checkIdentifier(identifier_)            \
  getToken();                                   \
  ASSERT_EQ_BINARY(tokenType::identifier,       \
                   token->type());              \
  ASSERT_EQ(identifier_,                        \
            ((identifierToken*) token)->value)

#define checkPrimitive(primitive_)                  \
  getToken();                                       \
  ASSERT_EQ_BINARY(tokenType::primitive,            \
                   token->type());                  \
  ASSERT_EQ(primitive_,                             \
            (int) ((primitiveToken*) token)->value)


  setStream("#pragma occa @tile(16, @outer, @inner)");
  // @
  checkOp(operatorType::attribute);
  // tile
  checkIdentifier("tile");
  // (
  checkOp(operatorType::parenthesesStart);
  // 16
  checkPrimitive(16)
    // ,
    checkOp(operatorType::comma);
  // @
  checkOp(operatorType::attribute);
  // outer
  checkIdentifier("outer");
  // ,
  checkOp(operatorType::comma);
  // @
  checkOp(operatorType::attribute);
  // inner
  checkIdentifier("inner");
  // )
  checkOp(operatorType::parenthesesEnd);

#undef checkOp
#undef checkIdentifier
#undef checkPrimitive
}
//======================================
