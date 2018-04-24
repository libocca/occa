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

#include "exprNode.hpp"
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

#define setStatement(str_, type_)                   \
  parseSource(str_);                                \
  OCCA_ASSERT_EQUAL(1,                              \
                    parser.root.size());            \
  OCCA_ASSERT_EQUAL_BINARY(type_,                   \
                           parser.root[0]->type())  \
  OCCA_ASSERT_TRUE(parser.success);                 \
  statement = parser.root[0]
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
  OCCA_ASSERT_FALSE(parser.isLoadingVariable())

vartype_t loadType(const std::string &s) {
  setSource(s);
  return parser.loadVariable().vartype;
}

#define assertVariable(str_)                            \
  setSource(str_);                                      \
  parser.preloadType();                                 \
  OCCA_ASSERT_FALSE(parser.isLoadingFunctionPointer()); \
  OCCA_ASSERT_TRUE(parser.isLoadingVariable())

variable loadVariable(const std::string &s) {
  setSource(s);
  return parser.loadVariable();
}

#define assertFunctionPointer(str_)                   \
  setSource(str_);                                    \
  parser.preloadType();                               \
  OCCA_ASSERT_TRUE(parser.isLoadingFunctionPointer())

void testBaseTypeLoading();
void testPointerTypeLoading();
void testReferenceTypeLoading();
void testArrayTypeLoading();
void testVariableLoading();
void testArgumentLoading();
void testFunctionPointerLoading();

void testTypeLoading() {
  testBaseTypeLoading();
  testPointerTypeLoading();
  testReferenceTypeLoading();
  testArrayTypeLoading();
  testVariableLoading();
  testArgumentLoading();
  testFunctionPointerLoading();
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

  std::cerr << "\n---[ Testing type loading errors ]--------------------\n\n";
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

  std::cerr << "\n---[ Testing type loading errors ]--------------------\n\n";
  type = preloadType("const *");
  type = preloadType("float * long");
}

void testReferenceTypeLoading() {
  vartype_t type;

  type = preloadType("int");
  OCCA_ASSERT_FALSE(type.isReference());
  type = preloadType("int &");
  OCCA_ASSERT_TRUE(type.isReference());

  type = preloadType("int *");
  OCCA_ASSERT_FALSE(type.isReference());
  type = preloadType("int *&");
  OCCA_ASSERT_TRUE(type.isReference());

  type = preloadType("int ***");
  OCCA_ASSERT_FALSE(type.isReference());
  type = preloadType("int ***&");
  OCCA_ASSERT_TRUE(type.isReference());
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

  std::cerr << "\n---[ Testing array type loading errors ]--------------------\n\n";
  assertType("int[-]");
  loadType("int[-]");
}

void testVariableLoading() {
  variable var;
  std::string varName;

  assertVariable("int varname[]");
  var = loadVariable("int varname[]");
  varName = var.name();
  OCCA_ASSERT_EQUAL("varname",
                    varName);
  OCCA_ASSERT_EQUAL(1,
                    (int) var.vartype.arrays.size());

  assertVariable("int varname[][]");
  var = loadVariable("int varname[][]");
  varName = var.name();
  OCCA_ASSERT_EQUAL("varname",
                    varName);
  OCCA_ASSERT_EQUAL(2,
                    (int) var.vartype.arrays.size());

  assertVariable("int varname[1]");
  var = loadVariable("int varname[1]");
  varName = var.name();
  OCCA_ASSERT_EQUAL("varname",
                    varName);
  OCCA_ASSERT_EQUAL(1,
                    (int) var.vartype.arrays.size());
  OCCA_ASSERT_EQUAL(1,
                    (int) var.vartype.arrays[0].evaluateSize());

  assertVariable("int varname[1 + 3][7]");
  var = loadVariable("int varname[1 + 3][7]");
  varName = var.name();
  OCCA_ASSERT_EQUAL("varname",
                    varName);
  OCCA_ASSERT_EQUAL(2,
                    (int) var.vartype.arrays.size());
  OCCA_ASSERT_EQUAL(4,
                    (int) var.vartype.arrays[0].evaluateSize());
  OCCA_ASSERT_EQUAL(7,
                    (int) var.vartype.arrays[1].evaluateSize());

  std::cerr << "\n---[ Testing variable loading errors ]--------------------\n\n";
  assertVariable("int varname[-]");
  loadVariable("int varname[-]");
}

void testArgumentLoading() {
  // Test argument detection
  tokenRangeVector argRanges;

  setSource("");
  parser.getArgumentRanges(argRanges);
  OCCA_ASSERT_EQUAL(0,
                    (int) argRanges.size());

  setSource("a, b");
  parser.getArgumentRanges(argRanges);
  OCCA_ASSERT_EQUAL(2,
                    (int) argRanges.size());

  setSource("(,,)");
  parser.getArgumentRanges(argRanges);
  OCCA_ASSERT_EQUAL(1,
                    (int) argRanges.size());

  setSource("(,,), (,,), (,,)");
  parser.getArgumentRanges(argRanges);
  OCCA_ASSERT_EQUAL(3,
                    (int) argRanges.size());

  // Removes trailing comma
  setSource("a, b,");
  parser.getArgumentRanges(argRanges);
  OCCA_ASSERT_EQUAL(2,
                    (int) argRanges.size());

  // Test arguments
}

void testFunctionPointerLoading() {
  variable var;
  std::string varName;
#define varFunc var.vartype.type->to<function_t>()

  // Test pointer vs block
  assertFunctionPointer("int (*varname)()");
  var = loadVariable("int (*varname)()");

  OCCA_ASSERT_EQUAL_BINARY(typeType::function,
                           var.vartype.type->type());
  varName = var.name();
  OCCA_ASSERT_EQUAL("varname",
                    varName);
  OCCA_ASSERT_TRUE(varFunc.isPointer);
  OCCA_ASSERT_FALSE(varFunc.isBlock);

  assertFunctionPointer("int (^varname)()");
  var = loadVariable("int (^varname)()");
  varName = var.name();
  OCCA_ASSERT_EQUAL("varname",
                    varName);
  OCCA_ASSERT_FALSE(varFunc.isPointer);
  OCCA_ASSERT_TRUE(varFunc.isBlock);

  // Test arguments
  var = loadVariable("int (*varname)()");
  OCCA_ASSERT_EQUAL(0,
                    (int) varFunc.args.size());

  var = loadVariable("int (*varname)(const int i = 0,)");
  OCCA_ASSERT_EQUAL(1,
                    (int) varFunc.args.size());
  OCCA_ASSERT_EQUAL(&int_,
                    varFunc.args[0].vartype.type);
  OCCA_ASSERT_TRUE(varFunc.args[0].vartype.has(const_));
  OCCA_ASSERT_EQUAL("i",
                    varFunc.args[0].name());

  var = loadVariable("int (*varname)(int, double,)");
  OCCA_ASSERT_EQUAL(2,
                    (int) varFunc.args.size());
  OCCA_ASSERT_EQUAL(&int_,
                    varFunc.args[0].vartype.type);
  OCCA_ASSERT_EQUAL(&double_,
                    varFunc.args[1].vartype.type);

#undef varFunc
}
//======================================

//---[ Loading ]------------------------
void testExpressionLoading();
void testDeclarationLoading();
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
void testBlockLoading();

void testLoading() {
  testExpressionLoading();
  testDeclarationLoading();
  testNamespaceLoading();
  // testTypeDeclLoading();
  testIfLoading();
  // testForLoading();
  testWhileLoading();
  testSwitchLoading();
  // testJumpsLoading();
  // testClassAccessLoading();
  // testAttributeLoading();
  // testPragmaLoading();
  // testGotoLoading();
  // testBlockLoading();
}

void testExpressionLoading() {
  statement_t *statement;
#define expr (*(statement->to<expressionStatement>().root))

  setStatement("2 + 3;",
               statementType::expression);
  OCCA_ASSERT_TRUE(expr.canEvaluate());
  OCCA_ASSERT_EQUAL(5,
                    (int) expr.evaluate());

  setStatement("-3;",
               statementType::expression);
  OCCA_ASSERT_TRUE(expr.canEvaluate());
  OCCA_ASSERT_EQUAL(-3,
                    (int) expr.evaluate());

  setStatement("sizeof(4);",
               statementType::expression);
  OCCA_ASSERT_TRUE(expr.canEvaluate());
  OCCA_ASSERT_EQUAL((uint64_t) sizeof(4),
                    (uint64_t) expr.evaluate());

  setStatement(";",
               statementType::expression);
  OCCA_ASSERT_FALSE(expr.canEvaluate());

#undef exprStatement
}

void testDeclarationLoading() {
  statement_t *statement;

#define decl statement->to<declarationStatement>()

  setStatement("int foo;",
               statementType::declaration);
  OCCA_ASSERT_EQUAL(1,
                    (int) decl.declarations.size());

  setStatement("int foo = 3;",
               statementType::declaration);
  OCCA_ASSERT_EQUAL(1,
                    (int) decl.declarations.size());
  OCCA_ASSERT_EQUAL(3,
                    (int) decl.declarations[0].value->evaluate());

  setStatement("int foo = 3, bar = 4;",
               statementType::declaration);
  OCCA_ASSERT_EQUAL(2,
                    (int) decl.declarations.size());
  OCCA_ASSERT_EQUAL(3,
                    (int) decl.declarations[0].value->evaluate());
  OCCA_ASSERT_EQUAL(4,
                    (int) decl.declarations[1].value->evaluate());

  setStatement("int foo = 3, *bar = 4;",
               statementType::declaration);
  OCCA_ASSERT_EQUAL(2,
                    (int) decl.declarations.size());
  OCCA_ASSERT_EQUAL(3,
                    (int) decl.declarations[0].value->evaluate());
  OCCA_ASSERT_EQUAL(4,
                    (int) decl.declarations[1].value->evaluate());

#undef decl
}

void testNamespaceLoading() {
  statement_t *statement;
  setStatement("namespace foo {}",
               statementType::namespace_);

  OCCA_ASSERT_EQUAL("foo",
                    statement->to<namespaceStatement>().name());

  setStatement("namespace A::B::C {}",
               statementType::namespace_);

  namespaceStatement &A = statement->to<namespaceStatement>();
  OCCA_ASSERT_EQUAL("A",
                    A.name());

  namespaceStatement &B = A[0]->to<namespaceStatement>();
  OCCA_ASSERT_EQUAL("B",
                    B.name());

  namespaceStatement &C = B[0]->to<namespaceStatement>();
  OCCA_ASSERT_EQUAL("C",
                    C.name());
}

void testTypeDeclLoading() {
  // statement_t *statement;
  // TODO: typedef
  // TODO: struct
  // TODO: enum
  // TODO: union
  // TODO: class
}

void testIfLoading() {
  statement_t *statement;

#define ifSmnt statement->to<ifStatement>()
#define condition (*ifSmnt.condition)
#define decl condition.to<declarationStatement>()

  setStatement("if (true) {}",
               statementType::if_);
  OCCA_ASSERT_EQUAL_BINARY(statementType::expression,
                           condition.type());
  OCCA_ASSERT_EQUAL(0,
                    (int) ifSmnt.elifSmnts.size());
  OCCA_ASSERT_FALSE(!!ifSmnt.elseSmnt);

  setStatement("if (true) {}\n"
               "else if (true) {}",
               statementType::if_);
  OCCA_ASSERT_EQUAL_BINARY(statementType::expression,
                           condition.type());
  OCCA_ASSERT_EQUAL(1,
                    (int) ifSmnt.elifSmnts.size());
  OCCA_ASSERT_FALSE(!!ifSmnt.elseSmnt);

  setStatement("if (true) {}\n"
               "else if (true) {}\n"
               "else if (true) {}",
               statementType::if_);
  OCCA_ASSERT_EQUAL_BINARY(statementType::expression,
                           condition.type());
  OCCA_ASSERT_EQUAL(2,
                    (int) ifSmnt.elifSmnts.size());
  OCCA_ASSERT_FALSE(!!ifSmnt.elseSmnt);

  setStatement("if (true) {}\n"
               "else if (true) {}\n"
               "else {}",
               statementType::if_);
  OCCA_ASSERT_EQUAL_BINARY(statementType::expression,
                           condition.type());
  OCCA_ASSERT_EQUAL(1,
                    (int) ifSmnt.elifSmnts.size());
  OCCA_ASSERT_TRUE(!!ifSmnt.elseSmnt);

  // Test declaration in conditional
  setStatement("if (const int i = 3) {}",
               statementType::if_);
  OCCA_ASSERT_EQUAL_BINARY(statementType::declaration,
                           condition.type());
  OCCA_ASSERT_EQUAL(1,
                    (int) decl.declarations.size());
  OCCA_ASSERT_EQUAL(3,
                    (int) decl.declarations[0].value->evaluate());

  // TODO: Test that 'i' exists in the if scope
#undef ifSmnt
#undef condition
#undef decl
}

void testForLoading() {
  statement_t *statement;

  setStatement("for (;;) {}",
               statementType::for_);
  setStatement("for (;;);",
               statementType::for_);

  // Test declaration in conditional
  setStatement("for (int i = 0; i < 2; ++i) {}",
               statementType::for_);

  // TODO: Test that 'i' exists in the if scope
}

void testWhileLoading() {
  statement_t *statement;

  setStatement("while (true) {}",
               statementType::while_);
  setStatement("while (true);",
               statementType::while_);

  // Test declaration in conditional
  setStatement("while (int i = 0) {}",
               statementType::while_);

  // TODO: Test that 'i' exists in the if scope

  // Same tests for do-while
  setStatement("do {} while (true);",
               statementType::while_);
  setStatement("do ; while (true);",
               statementType::while_);

  setStatement("do {} while (int i = 0);",
               statementType::while_);
}

void testSwitchLoading() {
  statement_t *statement;

#define switchSmnt statement->to<switchStatement>()
#define condition (*switchSmnt.condition)
#define decl condition.to<declarationStatement>()

  setStatement("switch (2) {}",
               statementType::switch_);
  OCCA_ASSERT_EQUAL_BINARY(statementType::expression,
                           condition.type());
  OCCA_ASSERT_EQUAL(0,
                    switchSmnt.size())

  // Weird cases
  setStatement("switch (2) case 2:;",
               statementType::switch_);
  OCCA_ASSERT_EQUAL_BINARY(statementType::expression,
                           condition.type());
  OCCA_ASSERT_EQUAL(2,
                    switchSmnt.size())

  setStatement("switch (2) case 2: 2;",
               statementType::switch_);
  OCCA_ASSERT_EQUAL_BINARY(statementType::expression,
                           condition.type());
  OCCA_ASSERT_EQUAL(2,
                    switchSmnt.size())

  // Test declaration in conditional
  setStatement("switch (int i = 2) {}",
               statementType::switch_);
  OCCA_ASSERT_EQUAL_BINARY(statementType::declaration,
                           condition.type());
  OCCA_ASSERT_EQUAL(0,
                    switchSmnt.size());
  OCCA_ASSERT_EQUAL(1,
                    (int) decl.declarations.size());
  OCCA_ASSERT_EQUAL(2,
                    (int) decl.declarations[0].value->evaluate());

  // TODO: Test that 'i' exists in the if scope

#undef switchSmnt
#undef condition
#undef decl
}

void testJumpsLoading() {
  statement_t *statement;

  setStatement("continue;",
               statementType::continue_);
  setStatement("break;",
               statementType::continue_);

  setStatement("return;",
               statementType::continue_);
  setStatement("return 1 + (2 * 1);",
               statementType::continue_);
  // TODO: Test 'eval' to make sure we capture the return value
}

void testClassAccessLoading() {
  statement_t *statement;

  setStatement("public:",
               statementType::classAccess);
  setStatement("protected:",
               statementType::classAccess);
  setStatement("private:",
               statementType::classAccess);
}

void testAttributeLoading() {
  statement_t *statement;

  setStatement("@dim",
               statementType::attribute);
  setStatement("@dim(2)",
               statementType::attribute);
  setStatement("@dim(x=2, y=2)",
               statementType::attribute);
  // TODO: Test the argument values
}

void testPragmaLoading() {
  statement_t *statement;

#define pragma_ statement->to<pragmaStatement>()

  setStatement("#pragma",
               statementType::pragma);
  OCCA_ASSERT_EQUAL("",
                    pragma_.token.value);

  setStatement("#pragma occa test",
               statementType::pragma);
  OCCA_ASSERT_EQUAL("occa test",
                    pragma_.token.value);

#undef pragma_
}

void testGotoLoading() {
  statement_t *statement;

  setStatement("label:",
               statementType::gotoLabel);
  setStatement("goto label;",
               statementType::goto_);
}

void testBlockLoading() {
  statement_t *statement;
  setStatement("{}",
               statementType::block);

  OCCA_ASSERT_EQUAL(0,
                    statement->to<blockStatement>().size());

  setStatement("{\n"
               " const int i = 0;\n"
               " ++i:\n"
               " namespace foo {}\n"
               " if (true) {}\n"
               "}\n",
               statementType::block);

  blockStatement &smnt = statement->to<blockStatement>();
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
//======================================

//---[ Errors ]------------------------
void testExpressionErrors();
void testDeclarationErrors();
void testNamespaceErrors();
void testTypeDeclErrors();
void testIfErrors();
void testForErrors();
void testWhileErrors();
void testSwitchErrors();
void testJumpsErrors();
void testClassAccessErrors();
void testAttributeErrors();
void testGotoErrors();

void testErrors() {
  std::cerr << "\n---[ Testing parser errors ]--------------------\n\n";

  testExpressionErrors();
  testDeclarationErrors();
  testNamespaceErrors();
  // testTypeDeclErrors();
  testIfErrors();
  // testForErrors();
  testWhileErrors();
  testSwitchErrors();
  // testJumpsErrors();
  // testClassAccessErrors();
  // testAttributeErrors();
  // testGotoErrors();
}

void testExpressionErrors() {
  parseSource("2 + 3");
  parseSource("-2");
  parseSource("2 = {}");
  parseSource("sizeof(4)");
}

void testDeclarationErrors() {
  parseSource("int foo");
  parseSource("int foo = 3");
  parseSource("int foo = 3, bar = 4");
  parseSource("int foo = 3, *bar = 4");
}

void testNamespaceErrors() {
  parseSource("namespace foo");
  parseSource("namespace foo::");
  parseSource("namespace foo::bar::");
  parseSource("namespace foo + {}");
}

void testTypeDeclErrors() {
}

void testIfErrors() {
  parseSource("if (true)");
  parseSource("if () {}");
  parseSource("if (if (true) {}) {}");
}

void testForErrors() {
}

void testWhileErrors() {
  parseSource("do {};");
  parseSource("do;");
  parseSource("do {} while (true)");
  parseSource("do ; while (true)");
  parseSource("do {} while (int i = 0)");
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
