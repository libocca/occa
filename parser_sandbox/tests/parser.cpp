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

#include "occa/tools/testing.hpp"

#include "parser.hpp"
#include "typeBuiltins.hpp"

void testTypeMethods();
void testPeek();
void testTypeLoading();
void testLoading();
void testErrors();

using namespace occa::lang;

//---[ Util Methods ]-------------------
std::string source;
parser_t parser;

void setSource(const std::string &s) {
  source = s;
  parser.setSource(s, false);
}

void parseSource(const std::string &s) {
  source = s;
  parser.parseSource(s);
}

template <class smntType>
smntType& getStatement(const int index = 0) {
  return parser.root[index]->to<smntType>();
}
//======================================

//---[ Macro Util Methods ]-------------
#define testStatementPeek(str_, type_)          \
  setSource(str_);                              \
  OCCA_ASSERT_EQUAL_BINARY(type_,               \
                           parser.peek());      \
  OCCA_ASSERT_TRUE(parser.success)

#define testStatementLoading(str_, type_)           \
  parseSource(str_);                                \
  OCCA_ASSERT_EQUAL(1,                              \
                    parser.root.size());            \
  OCCA_ASSERT_EQUAL_BINARY(type_,                   \
                           parser.root[0]->type())  \
  OCCA_ASSERT_TRUE(parser.success)
//======================================

int main(const int argc, const char **argv) {
  testTypeMethods();
  testPeek();
  testTypeLoading();
  testLoading();
  testErrors();

  return 0;
}

//---[ Utils ]--------------------------
void testTypeMethods() {
  setSource("int a = 0;");
  setSource("const int *a = 0;");

  // Make sure we can handle [long] and [long long]
  setSource("long a = 0;");
  setSource("const long a = 0;");

  setSource("long long a = 0;");
  setSource("const long long *a = 0;");
}
//======================================

//---[ Peek ]---------------------------
void testPeek() {
  testStatementPeek("",
                    statementType::none);

  testStatementPeek("#pragma",
                    statementType::pragma);
  testStatementPeek("#pragma occa test",
                    statementType::pragma);

  testStatementPeek("1 + 2;",
                    statementType::expression);
  testStatementPeek("\"a\";",
                    statementType::expression);
  testStatementPeek("'a';",
                    statementType::expression);
  testStatementPeek("sizeof 3;",
                    statementType::expression);

  testStatementPeek("int a = 0;",
                    statementType::declaration);
  testStatementPeek("const int a = 0;",
                    statementType::declaration);
  testStatementPeek("long long a, b = 0, *c = 0;",
                    statementType::declaration);

  testStatementPeek("goto foo;",
                    statementType::goto_);

  testStatementPeek("foo:",
                    statementType::gotoLabel);

  testStatementPeek("{}",
                    statementType::block);

  testStatementPeek("namespace foo {}",
                    statementType::namespace_);

  testStatementPeek("if (true) {}",
                    statementType::if_);
  testStatementPeek("else if (true) {}",
                    statementType::elif_);
  testStatementPeek("else {}",
                    statementType::else_);

  testStatementPeek("for () {}",
                    statementType::for_);

  testStatementPeek("while () {}",
                    statementType::while_);
  testStatementPeek("do {} while ()",
                    statementType::while_);

  testStatementPeek("switch () {}",
                    statementType::switch_);
  testStatementPeek("case foo:",
                    statementType::case_);
  testStatementPeek("continue;",
                    statementType::continue_);
  testStatementPeek("break;",
                    statementType::break_);

  testStatementPeek("return 0;",
                    statementType::return_);

  testStatementPeek("@attr",
                    statementType::attribute);
  testStatementPeek("@attr()",
                    statementType::attribute);

  testStatementPeek("public:",
                    statementType::classAccess);
  testStatementPeek("protected:",
                    statementType::classAccess);
  testStatementPeek("private:",
                    statementType::classAccess);
}
//======================================

//---[ Type Loading ]-------------------
vartype_t preloadType(const std::string &s) {
  setSource(s);
  return parser.preloadType();
}

#define assertType(str_)                                \
  setSource(str_);                                      \
  parser.preloadType();                                 \
  OCCA_ASSERT_FALSE(parser.isLoadingFunctionPointer()); \
  OCCA_ASSERT_FALSE(parser.isLoadingVariable());        \
  OCCA_ASSERT_TRUE(parser.isLoadingType())

vartype_t loadType(const std::string &s) {
  setSource(s);

  vartype_t vartype = parser.preloadType();
  parser.loadType(vartype);
  return vartype;
}

#define assertVariable(str_)                            \
  setSource(str_);                                      \
  parser.preloadType();                                 \
  OCCA_ASSERT_FALSE(parser.isLoadingFunctionPointer()); \
  OCCA_ASSERT_TRUE(parser.isLoadingVariable())

variable loadVariable(const std::string &s) {
  setSource(s);

  vartype_t vartype = parser.preloadType();
  return parser.loadVariable(vartype);
}

void testBaseTypeLoading();
void testPointerTypeLoading();
void testReferenceTypeLoading();
void testArrayTypeLoading();
void testVariableLoading();

void testTypeLoading() {
  testBaseTypeLoading();
  testPointerTypeLoading();
  testReferenceTypeLoading();
  testArrayTypeLoading();
  testVariableLoading();
}

void testBaseTypeLoading() {
  vartype_t type;

  // Test base type
  type = preloadType("int");
  OCCA_ASSERT_EQUAL(0,
                    type.qualifiers.size());
  OCCA_ASSERT_EQUAL(&int_,
                    type.type);

  type = preloadType("const volatile float");
  OCCA_ASSERT_EQUAL(2,
                    type.qualifiers.size());
  OCCA_ASSERT_TRUE(type.has(volatile_));
  OCCA_ASSERT_TRUE(type.has(const_));
  OCCA_ASSERT_EQUAL(&float_,
                    type.type);

  type = preloadType("const long long");
  OCCA_ASSERT_EQUAL(2,
                    type.qualifiers.size());
  OCCA_ASSERT_TRUE(type.has(const_));
  OCCA_ASSERT_TRUE(type.has(longlong_));
  OCCA_ASSERT_EQUAL(&int_,
                    type.type);

  // Test weird order declaration
  type = preloadType("double const long long");
  OCCA_ASSERT_EQUAL(2,
                    type.qualifiers.size());
  OCCA_ASSERT_TRUE(type.has(const_));
  OCCA_ASSERT_TRUE(type.has(longlong_));
  OCCA_ASSERT_EQUAL(&double_,
                    type.type);

  std::cerr << "Testing type loading errors:\n";
  type = preloadType("const");
  type = preloadType("const foo");
  type = preloadType("const const");
  type = preloadType("long long long");
}

void testPointerTypeLoading() {
  vartype_t type;

  type = preloadType("int *");
  OCCA_ASSERT_EQUAL(1,
                    (int) type.pointers.size());
  OCCA_ASSERT_EQUAL(0,
                    type.pointers[0].qualifiers.size());

  type = preloadType("const volatile float * const");
  OCCA_ASSERT_EQUAL(1,
                    (int) type.pointers.size());
  OCCA_ASSERT_EQUAL(1,
                    type.pointers[0].qualifiers.size());
  OCCA_ASSERT_TRUE(type.pointers[0].has(const_));

  type = preloadType("float * const * volatile ** const volatile restrict");
  OCCA_ASSERT_EQUAL(4,
                    (int) type.pointers.size());
  OCCA_ASSERT_TRUE(type.pointers[0].has(const_));
  OCCA_ASSERT_TRUE(type.pointers[1].has(volatile_));
  OCCA_ASSERT_EQUAL(0,
                    type.pointers[2].qualifiers.size());
  OCCA_ASSERT_TRUE(type.pointers[3].has(const_));
  OCCA_ASSERT_TRUE(type.pointers[3].has(volatile_));
  OCCA_ASSERT_TRUE(type.pointers[3].has(restrict_));

  std::cerr << "Testing type loading errors:\n";
  type = preloadType("const *");
  type = preloadType("float * long");
}

void testReferenceTypeLoading() {
  vartype_t type;

  type = preloadType("int");
  OCCA_ASSERT_FALSE(type.isReference);
  type = preloadType("int &");
  OCCA_ASSERT_TRUE(type.isReference);

  type = preloadType("int *");
  OCCA_ASSERT_FALSE(type.isReference);
  type = preloadType("int *&");
  OCCA_ASSERT_TRUE(type.isReference);

  type = preloadType("int ***");
  OCCA_ASSERT_FALSE(type.isReference);
  type = preloadType("int ***&");
  OCCA_ASSERT_TRUE(type.isReference);
}

void testArrayTypeLoading() {
  vartype_t type;

  assertType("int[]");
  type = loadType("int[]");
  OCCA_ASSERT_EQUAL(1,
                    (int) type.arrays.size());

  assertType("int[][]");
  type = loadType("int[][]");
  OCCA_ASSERT_EQUAL(2,
                    (int) type.arrays.size());

  assertType("int[1]");
  type = loadType("int[1]");
  OCCA_ASSERT_EQUAL(1,
                    (int) type.arrays.size());
  OCCA_ASSERT_EQUAL(1,
                    (int) type.arrays[0].evaluateSize());

  assertType("int[1 + 3][7]");
  type = loadType("int[1 + 3][7]");
  OCCA_ASSERT_EQUAL(2,
                    (int) type.arrays.size());
  OCCA_ASSERT_EQUAL(4,
                    (int) type.arrays[0].evaluateSize());
  OCCA_ASSERT_EQUAL(7,
                    (int) type.arrays[1].evaluateSize());

  std::cerr << "Testing array type loading errors:\n";
  assertType("int[-]");
  loadType("int[-]");
}

void testVariableLoading() {
  variable var;

  assertVariable("int varname[]");
  var = loadVariable("int varname[]");
  OCCA_ASSERT_EQUAL("varname",
                    var.name);
  OCCA_ASSERT_EQUAL(1,
                    (int) var.type.arrays.size());

  assertVariable("int varname[][]");
  var = loadVariable("int varname[][]");
  OCCA_ASSERT_EQUAL("varname",
                    var.name);
  OCCA_ASSERT_EQUAL(2,
                    (int) var.type.arrays.size());

  assertVariable("int varname[1]");
  var = loadVariable("int varname[1]");
  OCCA_ASSERT_EQUAL("varname",
                    var.name);
  OCCA_ASSERT_EQUAL(1,
                    (int) var.type.arrays.size());
  OCCA_ASSERT_EQUAL(1,
                    (int) var.type.arrays[0].evaluateSize());

  assertVariable("int varname[1 + 3][7]");
  var = loadVariable("int varname[1 + 3][7]");
  OCCA_ASSERT_EQUAL("varname",
                    var.name);
  OCCA_ASSERT_EQUAL(2,
                    (int) var.type.arrays.size());
  OCCA_ASSERT_EQUAL(4,
                    (int) var.type.arrays[0].evaluateSize());
  OCCA_ASSERT_EQUAL(7,
                    (int) var.type.arrays[1].evaluateSize());

  std::cerr << "Testing variable loading errors:\n";
  assertVariable("int varname[-]");
  loadVariable("int varname[-]");
}
//======================================

//---[ Loading ]------------------------
void testExpressionLoading();
void testDeclarationLoading();
void testBlockLoading();
void testNamespaceLoading();
void testTypeDeclLoading();
void testIfLoading();
void testForLoading();
void testWhileLoading();
void testSwitchLoading();
void testJumpsLoading();
void testClassAccessLoading();
void testAttributeLoading();
void testPragmaLoading();
void testGotoLoading();

void testLoading() {
  testExpressionLoading();
  testDeclarationLoading();
  // testBlockLoading();
  // testNamespaceLoading();
  // testTypeDeclLoading();
  // testIfLoading();
  // testForLoading();
  // testWhileLoading();
  // testSwitchLoading();
  // testJumpsLoading();
  // testClassAccessLoading();
  // testAttributeLoading();
  // testPragmaLoading();
  // testGotoLoading();
}

void testExpressionLoading() {
  testStatementLoading("2 + 3;",
                       statementType::expression);
  testStatementLoading("-3;",
                       statementType::expression);
  testStatementLoading(";",
                       statementType::expression);
  testStatementLoading("sizeof(4);",
                       statementType::expression);
  // TODO: Test we captured the proper expression by evaluating it
}

void testDeclarationLoading() {

}

void testBlockLoading() {
  testStatementLoading("{}",
                       statementType::block);

  OCCA_ASSERT_EQUAL(0,
                    getStatement<blockStatement>().size());

  testStatementLoading("{\n"
                       " const int i = 0;\n"
                       " ++i:\n"
                       " namespace foo {}\n"
                       " if (true) {}\n"
                       "}\n",
                       statementType::block);

  blockStatement &smnt = getStatement<blockStatement>();
  OCCA_ASSERT_EQUAL(4,
                    smnt.size());
  OCCA_ASSERT_EQUAL_BINARY(statementType::declaration,
                           smnt[0]->type());
  OCCA_ASSERT_EQUAL_BINARY(statementType::expression,
                           smnt[1]->type());
  OCCA_ASSERT_EQUAL_BINARY(statementType::namespace_,
                           smnt[2]->type());
  OCCA_ASSERT_EQUAL_BINARY(statementType::if_,
                           smnt[3]->type());
}

void testNamespaceLoading() {
  testStatementLoading("namespace foo {}",
                       statementType::namespace_);

  OCCA_ASSERT_EQUAL("foo",
                    getStatement<namespaceStatement>().name);

  testStatementLoading("namespace A::B::C {}",
                       statementType::namespace_);

  namespaceStatement &A = getStatement<namespaceStatement>();
  OCCA_ASSERT_EQUAL("A",
                    A.name);

  namespaceStatement &B = A[0]->to<namespaceStatement>();
  OCCA_ASSERT_EQUAL("B",
                    B.name);

  namespaceStatement &C = B[0]->to<namespaceStatement>();
  OCCA_ASSERT_EQUAL("C",
                    C.name);
}

void testTypeDeclLoading() {
  // TODO: typedef
  // TODO: struct
  // TODO: enum
  // TODO: union
  // TODO: class
}

void testIfLoading() {
  testStatementLoading("if (true) {}",
                       statementType::if_);

  testStatementLoading("if (true) {}\n"
                       "else if (true) {}",
                       statementType::if_);

  testStatementLoading("if (true) {}\n"
                       "else if (true) {}\n"
                       "else if (true) {}",
                       statementType::if_);

  testStatementLoading("if (true) {}\n"
                       "else if (true) {}\n"
                       "else {}",
                       statementType::if_);

  // Test declaration in conditional
  testStatementLoading("if (const int i = 1) {}",
                       statementType::if_);

  // TODO: Test that 'i' exists in the if scope
}

void testForLoading() {
  testStatementLoading("for (;;) {}",
                       statementType::for_);
  testStatementLoading("for (;;);",
                       statementType::for_);

  // Test declaration in conditional
  testStatementLoading("for (int i = 0; i < 2; ++i) {}",
                       statementType::for_);

  // TODO: Test that 'i' exists in the if scope
}

void testWhileLoading() {
  testStatementLoading("while (true) {}",
                       statementType::while_);
  testStatementLoading("while (true);",
                       statementType::while_);

  // Test declaration in conditional
  testStatementLoading("while (int i = 0) {}",
                       statementType::while_);

  // TODO: Test that 'i' exists in the if scope

  // Same tests for do-while
  testStatementLoading("do {} while (true);",
                       statementType::while_);
  testStatementLoading("do ; while (true);",
                       statementType::while_);

  testStatementLoading("do {} while (int i = 0)",
                       statementType::while_);
}

void testSwitchLoading() {
  testStatementLoading("switch (2) {}",
                       statementType::switch_);
  // Weird cases
  testStatementLoading("switch (2) case 2:",
                       statementType::switch_);
  testStatementLoading("switch (2) case 2: 2;",
                       statementType::switch_);

  // Test declaration in conditional
  testStatementLoading("switch (int i = 2) {}",
                       statementType::switch_);

  // TODO: Test that 'i' exists in the if scope

  // Test caseStatement
  testStatementLoading("case 2:",
                       statementType::case_);
  testStatementLoading("case 2: 2;",
                       statementType::case_);

  // Test defaultStatement
  testStatementLoading("default:",
                       statementType::default_);
  testStatementLoading("default: 2;",
                       statementType::default_);
}

void testJumpsLoading() {
  testStatementLoading("continue;",
                       statementType::continue_);
  testStatementLoading("break;",
                       statementType::continue_);

  testStatementLoading("return;",
                       statementType::continue_);
  testStatementLoading("return 1 + (2 * 1);",
                       statementType::continue_);
  // TODO: Test 'eval' to make sure we capture the return value
}

void testClassAccessLoading() {
  testStatementLoading("public:",
                       statementType::classAccess);
  testStatementLoading("protected:",
                       statementType::classAccess);
  testStatementLoading("private:",
                       statementType::classAccess);
}

void testAttributeLoading() {
  testStatementLoading("@dim",
                       statementType::attribute);
  testStatementLoading("@dim(2)",
                       statementType::attribute);
  testStatementLoading("@dim(x=2, y=2)",
                       statementType::attribute);
  // TODO: Test the argument values
}

void testPragmaLoading() {
  testStatementLoading("#pragma",
                       statementType::pragma);
  testStatementLoading("#pragma occa test",
                       statementType::pragma);
  // TODO: Test the pragma source
}

void testGotoLoading() {
  testStatementLoading("label:",
                       statementType::gotoLabel);
  testStatementLoading("goto label;",
                       statementType::goto_);
}
//======================================

//---[ Errors ]------------------------
void testExpressionErrors();
void testDeclarationErrors();
void testBlockErrors();
void testNamespaceErrors();
void testTypeDeclErrors();
void testIfErrors();
void testForErrors();
void testWhileErrors();
void testSwitchErrors();
void testJumpsErrors();
void testClassAccessErrors();
void testAttributeErrors();
void testPragmaErrors();
void testGotoErrors();

void testErrors() {
  std::cerr << "Testing parser errors:\n";

  testExpressionErrors();
  testDeclarationErrors();
  // testBlockErrors();
  // testNamespaceErrors();
  // testTypeDeclErrors();
  // testIfErrors();
  // testForErrors();
  // testWhileErrors();
  // testSwitchErrors();
  // testJumpsErrors();
  // testClassAccessErrors();
  // testAttributeErrors();
  // testPragmaErrors();
  // testGotoErrors();
}

void testExpressionErrors() {
  parseSource("2 + 3");
  parseSource("-2");
  parseSource("2 = {}");
  parseSource("sizeof(4)");
}

void testDeclarationErrors() {
}

void testBlockErrors() {
}

void testNamespaceErrors() {
}

void testTypeDeclErrors() {
}

void testIfErrors() {
}

void testForErrors() {
}

void testWhileErrors() {
}

void testSwitchErrors() {
}

void testJumpsErrors() {
}

void testClassAccessErrors() {
}

void testAttributeErrors() {
}

void testPragmaErrors() {
}

void testGotoErrors() {
}
//======================================
