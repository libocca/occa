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

#include "parserUtils.hpp"

void testTypeMethods();
void testPeek();
void testTypeLoading();
void testTypeErrors();
void testLoading();
void testErrors();
void testScope();

int main(const int argc, const char **argv) {
  parser.addAttribute<dummy>();
  parser.addAttribute<attributes::kernel>();
  parser.addAttribute<attributes::outer>();
  parser.addAttribute<attributes::inner>();
  parser.addAttribute<attributes::shared>();
  parser.addAttribute<attributes::exclusive>();

  testTypeMethods();
  testPeek();

  testTypeLoading();
  testTypeErrors();

  testLoading();
  testErrors();

  testScope();

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

  testStatementPeek(";",
                    statementType::empty);

  testStatementPeek("#pragma",
                    statementType::pragma);
  testStatementPeek("#pragma occa @dim(5)\n"
                    "int *x;",
                    statementType::declaration);

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
  testStatementPeek("default:",
                    statementType::default_);
  testStatementPeek("continue;",
                    statementType::continue_);
  testStatementPeek("break;",
                    statementType::break_);

  testStatementPeek("return 0;",
                    statementType::return_);

  testStatementPeek("public:",
                    statementType::classAccess);
  testStatementPeek("protected:",
                    statementType::classAccess);
  testStatementPeek("private:",
                    statementType::classAccess);
}
//======================================

//---[ Type Loading ]-------------------
vartype_t loadType(const std::string &s) {
  setSource(s);
  return parser.loadType();
}

#define assertType(str_)                            \
  setSource(str_);                                  \
  parser.loadType();                                \
  ASSERT_FALSE(parser.isLoadingFunctionPointer());  \
  ASSERT_FALSE(parser.isLoadingVariable())

vartype_t loadVariableType(const std::string &s) {
  setSource(s);
  return parser.loadVariable().vartype;
}

#define assertVariable(str_)                        \
  setSource(str_);                                  \
  parser.loadType();                                \
  ASSERT_FALSE(parser.isLoadingFunctionPointer());  \
  ASSERT_TRUE(parser.isLoadingVariable())

variable_t loadVariable(const std::string &s) {
  setSource(s);
  return parser.loadVariable();
}

#define assertFunctionPointer(str_)               \
  setSource(str_);                                \
  parser.loadType();                              \
  ASSERT_TRUE(parser.isLoadingFunctionPointer())

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
  type = loadType("int");
  ASSERT_EQ(0,
            type.qualifiers.size());
  ASSERT_EQ(&int_,
            type.type);

  type = loadType("const volatile float");
  ASSERT_EQ(2,
            type.qualifiers.size());
  ASSERT_TRUE(type.has(volatile_));
  ASSERT_TRUE(type.has(const_));
  ASSERT_EQ(&float_,
            type.type);

  type = loadType("const long long");
  ASSERT_EQ(2,
            type.qualifiers.size());
  ASSERT_TRUE(type.has(const_));
  ASSERT_TRUE(type.has(longlong_));
  ASSERT_EQ(&int_,
            type.type);

  // Test weird order declaration
  type = loadType("double const long long");
  ASSERT_EQ(2,
            type.qualifiers.size());
  ASSERT_TRUE(type.has(const_));
  ASSERT_TRUE(type.has(longlong_));
  ASSERT_EQ(&double_,
            type.type);
}

void testPointerTypeLoading() {
  vartype_t type;

  type = loadType("int *");
  ASSERT_EQ(1,
            (int) type.pointers.size());
  ASSERT_EQ(0,
            type.pointers[0].qualifiers.size());

  type = loadType("const volatile float * const");
  ASSERT_EQ(1,
            (int) type.pointers.size());
  ASSERT_EQ(1,
            type.pointers[0].qualifiers.size());
  ASSERT_TRUE(type.pointers[0].has(const_));

  type = loadType("float * const * volatile ** const volatile");
  ASSERT_EQ(4,
            (int) type.pointers.size());
  ASSERT_TRUE(type.pointers[0].has(const_));
  ASSERT_TRUE(type.pointers[1].has(volatile_));
  ASSERT_EQ(0,
            type.pointers[2].qualifiers.size());
  ASSERT_TRUE(type.pointers[3].has(const_));
  ASSERT_TRUE(type.pointers[3].has(volatile_));
}

void testReferenceTypeLoading() {
  vartype_t type;

  type = loadType("int");
  ASSERT_FALSE(type.isReference());
  type = loadType("int &");
  ASSERT_TRUE(type.isReference());

  type = loadType("int *");
  ASSERT_FALSE(type.isReference());
  type = loadType("int *&");
  ASSERT_TRUE(type.isReference());

  type = loadType("int ***");
  ASSERT_FALSE(type.isReference());
  type = loadType("int ***&");
  ASSERT_TRUE(type.isReference());
}

void testArrayTypeLoading() {
  vartype_t type;

  assertType("int[]");
  type = loadVariableType("int[]");
  ASSERT_EQ(1,
            (int) type.arrays.size());

  assertType("int[][]");
  type = loadVariableType("int[][]");
  ASSERT_EQ(2,
            (int) type.arrays.size());

  assertType("int[1]");
  type = loadVariableType("int[1]");
  ASSERT_EQ(1,
            (int) type.arrays.size());
  ASSERT_EQ(1,
            (int) type.arrays[0].evaluateSize());

  assertType("int[1 + 3][7]");
  type = loadVariableType("int[1 + 3][7]");
  ASSERT_EQ(2,
            (int) type.arrays.size());
  ASSERT_EQ(4,
            (int) type.arrays[0].evaluateSize());
  ASSERT_EQ(7,
            (int) type.arrays[1].evaluateSize());
}

void testVariableLoading() {
  variable_t var;
  std::string varName;

  assertVariable("int varname[]");
  var = loadVariable("int varname[]");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_EQ(1,
            (int) var.vartype.arrays.size());

  assertVariable("int varname[][]");
  var = loadVariable("int varname[][]");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_EQ(2,
            (int) var.vartype.arrays.size());

  assertVariable("int varname[1]");
  var = loadVariable("int varname[1]");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_EQ(1,
            (int) var.vartype.arrays.size());
  ASSERT_EQ(1,
            (int) var.vartype.arrays[0].evaluateSize());

  assertVariable("int varname[1 + 3][7]");
  var = loadVariable("int varname[1 + 3][7]");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_EQ(2,
            (int) var.vartype.arrays.size());
  ASSERT_EQ(4,
            (int) var.vartype.arrays[0].evaluateSize());
  ASSERT_EQ(7,
            (int) var.vartype.arrays[1].evaluateSize());
}

void testArgumentLoading() {
  // Test argument detection
  tokenRangeVector argRanges;

  setSource("");
  parser.getArgumentRanges(argRanges);
  ASSERT_EQ(0,
            (int) argRanges.size());

  setSource("a, b");
  parser.getArgumentRanges(argRanges);
  ASSERT_EQ(2,
            (int) argRanges.size());

  setSource("(,,)");
  parser.getArgumentRanges(argRanges);
  ASSERT_EQ(1,
            (int) argRanges.size());

  setSource("(,,), (,,), (,,)");
  parser.getArgumentRanges(argRanges);
  ASSERT_EQ(3,
            (int) argRanges.size());

  // Removes trailing comma
  setSource("a, b,");
  parser.getArgumentRanges(argRanges);
  ASSERT_EQ(2,
            (int) argRanges.size());

  // Test arguments
}

void testFunctionPointerLoading() {
  variable_t var;
  std::string varName;

#define varFunc var.vartype.type->to<functionPtr_t>()

  // Test pointer vs block
  assertFunctionPointer("int (*varname)()");
  var = loadVariable("int (*varname)()");

  ASSERT_EQ_BINARY(typeType::functionPtr,
                   var.vartype.type->type());
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_FALSE(varFunc.isBlock);

  assertFunctionPointer("int (^varname)()");
  var = loadVariable("int (^varname)()");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_TRUE(varFunc.isBlock);

  // Test arguments
  var = loadVariable("int (*varname)()");
  ASSERT_EQ(0,
            (int) varFunc.args.size());

  var = loadVariable("int (*varname)(const int i = 0,)");
  ASSERT_EQ(1,
            (int) varFunc.args.size());
  ASSERT_EQ(&int_,
            varFunc.args[0].vartype.type);
  ASSERT_TRUE(varFunc.args[0].vartype.has(const_));
  ASSERT_EQ("i",
            varFunc.args[0].name());

  var = loadVariable("int (*varname)(int, double,)");
  ASSERT_EQ(2,
            (int) varFunc.args.size());
  ASSERT_EQ(&int_,
            varFunc.args[0].vartype.type);
  ASSERT_EQ(&double_,
            varFunc.args[1].vartype.type);

#undef varFunc
}

void testBaseTypeErrors();
void testPointerTypeErrors();
void testArrayTypeErrors();
void testVariableErrors();

void testTypeErrors() {
  std::cerr << "\n---[ Testing type errors ]----------------------\n\n";
  testBaseTypeErrors();
  testPointerTypeErrors();
  testArrayTypeErrors();
  testVariableErrors();
  std::cerr << "================================================\n\n";
}

void testBaseTypeErrors() {
  vartype_t type;
  type = loadType("const");
  type = loadType("const foo");
  type = loadType("const const");
  type = loadType("long long long");
}

void testPointerTypeErrors() {
  vartype_t type;
  type = loadType("const *");
  type = loadType("float * long");
}

void testArrayTypeErrors() {
  assertType("int[-]");
  loadVariableType("int[-]");
}

void testVariableErrors() {
  assertVariable("int varname[-]");
  loadVariable("int varname[-]");
}
//======================================

//---[ Loading ]------------------------
void testExpressionLoading();
void testDeclarationLoading();
void testNamespaceLoading();
void testTypeDeclLoading();
void testFunctionLoading();
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
  testFunctionLoading();
  testIfLoading();
  testForLoading();
  testWhileLoading();
  testSwitchLoading();
  testJumpsLoading();
  testClassAccessLoading();
  testPragmaLoading();
  testGotoLoading();
  testBlockLoading();
  testAttributeLoading();
}

void testExpressionLoading() {
  statement_t *statement;
#define expr (*(statement->to<expressionStatement>().expr))

  setStatement("2 + 3;",
               statementType::expression);
  ASSERT_TRUE(expr.canEvaluate());
  ASSERT_EQ(5,
            (int) expr.evaluate());

  setStatement("-3;",
               statementType::expression);
  ASSERT_TRUE(expr.canEvaluate());
  ASSERT_EQ(-3,
            (int) expr.evaluate());

  setStatement("sizeof(4);",
               statementType::expression);
  ASSERT_TRUE(expr.canEvaluate());
  ASSERT_EQ((uint64_t) sizeof(4),
            (uint64_t) expr.evaluate());

  parseAndPrintSource("a[i] = b >= 0 ? c[i] : -d[-e - 1];");

#undef expr
}

void testDeclarationLoading() {
  statement_t *statement;

#define decl         statement->to<declarationStatement>()
#define decls        decl.declarations
#define declVar(N)   (*decls[N].variable)
#define declValue(N) (*(decls[N].value))

  setStatement("int foo;",
               statementType::declaration);
  ASSERT_EQ(1,
            (int) decls.size());

  setStatement("int foo = 3;",
               statementType::declaration);
  ASSERT_EQ(1,
            (int) decls.size());
  ASSERT_EQ("foo",
            declVar(0).name());
  ASSERT_EQ(3,
            (int) declValue(0).evaluate());

  setStatement("int foo = 3, bar = 4;",
               statementType::declaration);
  ASSERT_EQ(2,
            (int) decls.size());
  ASSERT_EQ("foo",
            declVar(0).name());
  ASSERT_EQ(3,
            (int) declValue(0).evaluate());
  ASSERT_EQ("bar",
            declVar(1).name());
  ASSERT_EQ(4,
            (int) declValue(1).evaluate());

  setStatement("int foo = 3, *bar = 4;",
               statementType::declaration);
  ASSERT_EQ(2,
            (int) decls.size());
  ASSERT_EQ("foo",
            declVar(0).name());
  ASSERT_EQ(0,
            (int) declVar(0).vartype.pointers.size());
  ASSERT_EQ(3,
            (int) declValue(0).evaluate());
  ASSERT_EQ("bar",
            declVar(1).name());
  ASSERT_EQ(1,
            (int) declVar(1).vartype.pointers.size());
  ASSERT_EQ(4,
            (int) declValue(1).evaluate());

  setStatement("int *foo = 3, bar = 4;",
               statementType::declaration);
  ASSERT_EQ(2,
            (int) decls.size());
  ASSERT_EQ("foo",
            declVar(0).name());
  ASSERT_EQ(1,
            (int) declVar(0).vartype.pointers.size());
  ASSERT_EQ(3,
            (int) declValue(0).evaluate());
  ASSERT_EQ("bar",
            declVar(1).name());
  ASSERT_EQ(0,
            (int) declVar(1).vartype.pointers.size());
  ASSERT_EQ(4,
            (int) declValue(1).evaluate());

  setStatement("int foo { 3 }, *bar { 4 };",
               statementType::declaration);
  ASSERT_EQ(2,
            (int) decls.size());
  ASSERT_EQ("foo",
            declVar(0).name());
  ASSERT_EQ(3,
            (int) declValue(0).evaluate());
  ASSERT_EQ("bar",
            declVar(1).name());
  ASSERT_EQ(4,
            (int) declValue(1).evaluate());

  setStatement("int foo : 1 = 3, bar : 2 = 4;",
               statementType::declaration);
  ASSERT_EQ(2,
            (int) decls.size());
  ASSERT_EQ("foo",
            declVar(0).name());
  ASSERT_EQ(1,
            declVar(0).vartype.bitfield);
  ASSERT_EQ("bar",
            declVar(1).name());
  ASSERT_EQ(2,
            declVar(1).vartype.bitfield);

#undef decl
#undef decls
#undef declVar
#undef declValue
}

void testNamespaceLoading() {
  statement_t *statement;
  setStatement("namespace foo {}",
               statementType::namespace_);

  ASSERT_EQ("foo",
            statement->to<namespaceStatement>().name());

  setStatement("namespace A::B::C {}",
               statementType::namespace_);

  namespaceStatement &A = statement->to<namespaceStatement>();
  ASSERT_EQ("A",
            A.name());

  namespaceStatement &B = A[0]->to<namespaceStatement>();
  ASSERT_EQ("B",
            B.name());

  namespaceStatement &C = B[0]->to<namespaceStatement>();
  ASSERT_EQ("C",
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

void testFunctionLoading() {
  statement_t *statement;

#define funcSmnt     statement->to<functionStatement>()
#define funcDeclSmnt statement->to<functionDeclStatement>()
#define func         funcSmnt.function
#define funcDecl     funcDeclSmnt.function

  setStatement("void foo();",
               statementType::function);
  ASSERT_EQ("foo",
            func.name());
  ASSERT_EQ(&void_,
            func.returnType.type);

  setStatement("int* bar();",
               statementType::function);
  ASSERT_EQ("bar",
            func.name());
  ASSERT_EQ(&int_,
            func.returnType.type);
  ASSERT_EQ(1,
            (int) func.returnType.pointers.size());

  setStatement("void foo2(int a) {}",
               statementType::functionDecl);
  ASSERT_EQ("foo2",
            funcDecl.name());
  ASSERT_EQ(&void_,
            funcDecl.returnType.type);
  ASSERT_EQ(1,
            (int) funcDecl.args.size());
  ASSERT_EQ(0,
            funcDeclSmnt.size());

  setStatement("void foo3(int a, int b) { int x; int y; }",
               statementType::functionDecl);
  ASSERT_EQ("foo3",
            funcDecl.name());
  ASSERT_EQ(&void_,
            funcDecl.returnType.type);
  ASSERT_EQ(2,
            (int) funcDecl.args.size());
  ASSERT_EQ(2,
            funcDeclSmnt.size());

#undef funcSmnt
#undef func
}

void testIfLoading() {
  statement_t *statement;

#define ifSmnt       statement->to<ifStatement>()
#define condition    (*ifSmnt.condition)
#define decl         condition.to<declarationStatement>()
#define decls        decl.declarations
#define declVar(N)   (*decls[N].variable)
#define declValue(N) (*(decls[N].value))

  setStatement("if (true) {}",
               statementType::if_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(0,
            (int) ifSmnt.elifSmnts.size());
  ASSERT_FALSE(!!ifSmnt.elseSmnt);

  setStatement("if (true) {}\n"
               "else if (true) {}",
               statementType::if_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(1,
            (int) ifSmnt.elifSmnts.size());
  ASSERT_FALSE(!!ifSmnt.elseSmnt);

  setStatement("if (true) {}\n"
               "else if (true) {}\n"
               "else if (true) {}",
               statementType::if_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(2,
            (int) ifSmnt.elifSmnts.size());
  ASSERT_FALSE(!!ifSmnt.elseSmnt);

  setStatement("if (true) {}\n"
               "else if (true) {}\n"
               "else {}",
               statementType::if_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(1,
            (int) ifSmnt.elifSmnts.size());
  ASSERT_TRUE(!!ifSmnt.elseSmnt);

  // Test declaration in conditional
  setStatement("if (const int i = 3) {}",
               statementType::if_);
  ASSERT_EQ_BINARY(statementType::declaration,
                   condition.type());
  ASSERT_EQ(1,
            (int) decls.size());
  ASSERT_EQ("i",
            declVar(0).name());
  ASSERT_EQ(3,
            (int) declValue(0).evaluate());

  // TODO: Test that 'i' exists in the if scope

#undef ifSmnt
#undef condition
#undef decl
#undef decls
#undef declVar
#undef declValue
}

void testForLoading() {
  statement_t *statement;

#define forSmnt statement->to<forStatement>()
#define init (*forSmnt.init)
#define check (*forSmnt.check)
#define update (*forSmnt.update)
#define initDecl init.to<declarationStatement>()

  setStatement("for (;;) {}",
               statementType::for_);
  ASSERT_EQ_BINARY(statementType::empty,
                   init.type());
  ASSERT_EQ_BINARY(statementType::empty,
                   check.type());
  ASSERT_EQ_BINARY(statementType::empty,
                   update.type());
  ASSERT_EQ(0,
            (int) forSmnt.children.size());

  setStatement("for (;;);",
               statementType::for_);
  ASSERT_EQ_BINARY(statementType::empty,
                   init.type());
  ASSERT_EQ_BINARY(statementType::empty,
                   check.type());
  ASSERT_EQ_BINARY(statementType::empty,
                   update.type());
  ASSERT_EQ(1,
            (int) forSmnt.children.size());

  // Test declaration in conditional
  setStatement("for (int i = 0; i < 2; ++i) {}",
               statementType::for_);
  ASSERT_EQ_BINARY(statementType::declaration,
                   init.type());
  ASSERT_EQ_BINARY(statementType::expression,
                   check.type());
  ASSERT_EQ_BINARY(statementType::expression,
                   update.type());

  // TODO: Test that 'i' exists in the if scope

#undef forSmnt
#undef init
#undef check
#undef update
#undef initDecl
}

void testWhileLoading() {
  statement_t *statement;
  (void) statement;

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
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(0,
            switchSmnt.size())

    // Weird cases
    setStatement("switch (2) ;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(1,
            switchSmnt.size())

    // Weird cases
    setStatement("switch (2) case 2:;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(2,
            switchSmnt.size())

    setStatement("switch (2) default:;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(2,
            switchSmnt.size())

    setStatement("switch (2) case 2: 2;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(2,
            switchSmnt.size())

    setStatement("switch (2) default: 2;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(2,
            switchSmnt.size())

    // Test declaration in conditional
    setStatement("switch (int i = 2) {}",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::declaration,
                   condition.type());
  ASSERT_EQ(0,
            switchSmnt.size());
  ASSERT_EQ(1,
            (int) decl.declarations.size());
  ASSERT_EQ(2,
            (int) decl.declarations[0].value->evaluate());

  // TODO: Test that 'i' exists in the if scope

#undef switchSmnt
#undef condition
#undef decl
}

void testJumpsLoading() {
  statement_t *statement;

#define returnValue statement->to<returnStatement>().value

  setStatement("continue;",
               statementType::continue_);

  setStatement("break;",
               statementType::break_);

  setStatement("return;",
               statementType::return_);
  ASSERT_EQ((void*) NULL,
            (void*) returnValue);

  setStatement("return 1 + (2 * 3);",
               statementType::return_);
  ASSERT_EQ(7,
            (int) returnValue->evaluate());

#undef returnValue
}

void testClassAccessLoading() {
  statement_t *statement;

#define access statement->to<classAccessStatement>().access

  setStatement("public:",
               statementType::classAccess);
  ASSERT_EQ(classAccess::public_,
            access);

  setStatement("protected:",
               statementType::classAccess);
  ASSERT_EQ(classAccess::protected_,
            access);

  setStatement("private:",
               statementType::classAccess);
  ASSERT_EQ(classAccess::private_,
            access);

#undef access
}

void testPragmaLoading() {
  statement_t *statement;

#define pragma_ statement->to<pragmaStatement>()

  setStatement("#pragma",
               statementType::pragma);
  ASSERT_EQ("",
            pragma_.token.value);

  setStatement("#pragma omp parallel for",
               statementType::pragma);
  ASSERT_EQ("omp parallel for",
            pragma_.token.value);

#undef pragma_
}

void testGotoLoading() {
  statement_t *statement;
  (void) statement;

  setStatement("label:",
               statementType::gotoLabel);
  setStatement("goto label;",
               statementType::goto_);
}

void testBlockLoading() {
  statement_t *statement;
  setStatement("{}",
               statementType::block);

  ASSERT_EQ(0,
            statement->to<blockStatement>().size());

  setStatement("{\n"
               " const int i = 0;\n"
               " ++i;\n"
               " if (true) {}\n"
               " if (true) {} else {}\n"
               " while (true) {}\n"
               " do {} while (true);\n"
               " switch (1) default:;\n"
               "}\n",
               statementType::block);

  blockStatement &smnt = statement->to<blockStatement>();
  ASSERT_EQ(7,
            smnt.size());
  ASSERT_EQ_BINARY(statementType::declaration,
                   smnt[0]->type());
  ASSERT_EQ_BINARY(statementType::expression,
                   smnt[1]->type());
  ASSERT_EQ_BINARY(statementType::if_,
                   smnt[2]->type());
  ASSERT_EQ_BINARY(statementType::if_,
                   smnt[3]->type());
  ASSERT_EQ_BINARY(statementType::while_,
                   smnt[4]->type());
  ASSERT_EQ_BINARY(statementType::while_,
                   smnt[5]->type());
  ASSERT_EQ_BINARY(statementType::switch_,
                   smnt[6]->type());
}

void testAttributeLoading() {
  statement_t *statement;

#define smntAttr(N)       statement->attributes[N]->name()
#define declSmnt          statement->to<declarationStatement>()
#define decls             declSmnt.declarations
#define declVar(N)        (*decls[N].variable)
#define declVarAttr(N, A) declVar(N).attributes[A]

  setStatement("const int *x @dim(2, 3), *y;",
               statementType::declaration);
  ASSERT_EQ(0,
            (int) statement->attributes.size());
  ASSERT_EQ(1,
            (int) declVar(0).attributes.size());
  ASSERT_EQ("dim",
            declVarAttr(0, "dim").name());
  ASSERT_EQ(0,
            (int) declVar(1).attributes.size());

  attributeToken_t &xDim1 = declVarAttr(0, "dim");
  ASSERT_EQ(2,
            (int) xDim1.args.size());
  ASSERT_EQ(2,
            (int) xDim1[0]->expr->evaluate());
  ASSERT_EQ(3,
            (int) xDim1[1]->expr->evaluate());

  setStatement("const int *x @dummy(x=2, y=3), *y;",
               statementType::declaration);
  ASSERT_EQ(0,
            (int) statement->attributes.size());
  ASSERT_EQ(1,
            (int) declVar(0).attributes.size());
  ASSERT_EQ("dummy",
            declVarAttr(0, "dummy").name());
  ASSERT_EQ(0,
            (int) declVar(1).attributes.size());

  attributeToken_t &xDummy = declVarAttr(0, "dummy");
  ASSERT_EQ(2,
            (int) xDummy.kwargs.size());
  ASSERT_EQ(2,
            (int) xDummy["x"]->expr->evaluate());
  ASSERT_EQ(3,
            (int) xDummy["y"]->expr->evaluate());

  setStatement("@dim(2 + 2, 10 - 5) const int *x, *y;",
               statementType::declaration);
  ASSERT_EQ(1,
            (int) statement->attributes.size());
  ASSERT_EQ(1,
            (int) declVar(0).attributes.size());
  ASSERT_EQ("dim",
            declVarAttr(0, "dim").name());
  ASSERT_EQ(1,
            (int) declVar(1).attributes.size());
  ASSERT_EQ("dim",
            declVarAttr(1, "dim").name());

  attributeToken_t &xDim3 = declVarAttr(0, "dim");
  ASSERT_EQ(2,
            (int) xDim3.args.size());
  ASSERT_EQ(4,
            (int) xDim3[0]->expr->evaluate());
  ASSERT_EQ(5,
            (int) xDim3[1]->expr->evaluate());

  attributeToken_t &xDim4 = declVarAttr(1, "dim");
  ASSERT_EQ(2,
            (int) xDim4.args.size());
  ASSERT_EQ(4,
            (int) xDim4[0]->expr->evaluate());
  ASSERT_EQ(5,
            (int) xDim4[1]->expr->evaluate());

  std::cerr << "\n---[ @dim Transformations ]---------------------\n";
  parseAndPrintSource("@dim(1,2,3) int *x; x(1,2,3);");
  parseAndPrintSource("@dim(3,2,1) int *x; x(1,2,3);");
  parseAndPrintSource("@dim(1,2,3) @dimOrder(0,1,2) int *x; x(1,2,3);");
  parseAndPrintSource("@dim(1,2,3) @dimOrder(1,2,0) int *x; x(1,2,3);");
  parseAndPrintSource("@dim(1,2,3) @dimOrder(2,0,1) int *x; x(1,2,3);");
  parseAndPrintSource("@dim(1,2,3) @dimOrder(2,1,0) int *x; x(1,2,3);");
  std::cerr << "==============================================\n";

  std::cerr << "\n---[ @tile Transformations ]--------------------\n";
  parseAndPrintSource("for (int i = 0; i < (1 + 2 + N + 6); ++i; @tile(16, @outer, @inner, check=false)) {"
                      "  int x;"
                      "}");
  parseAndPrintSource("for (int i = 0; i > (1 + 2 + N + 6); --i; @tile(16, @outer, @inner, check=false)) {"
                      "  int x;"
                      "}");
  parseAndPrintSource("for (int i = 0; i <= (1 + 2 + N + 6); i++; @tile(16, @outer, @inner, check=false)) {"
                      "  int x;"
                      "}");
  parseAndPrintSource("for (int i = 0; i >= (1 + 2 + N + 6); i--; @tile(16, @outer, @inner, check=true)) {"
                      "  int x;"
                      "}");
  parseAndPrintSource("for (int i = 0; (1 + 2 + N + 6) > i; i += 3; @tile(16, @outer, @inner, check=true)) {"
                      "  int x;"
                      "}");
  parseAndPrintSource("for (int i = 0; (1 + 2 + N + 6) <= i; i -= 2 + 3; @tile(16, @outer, @inner, check=true)) {"
                      "  int x;"
                      "}");
  std::cerr << "==============================================\n\n";

#undef smntAttr
#undef declSmnt
#undef decls
#undef declVar
#undef declVarAttr
}
//======================================

//---[ Errors ]------------------------
void testExpressionErrors();
void testDeclarationErrors();
void testNamespaceErrors();
void testTypeDeclErrors();
void testFunctionErrors();
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
  testFunctionErrors();
  testIfErrors();
  testForErrors();
  testWhileErrors();
  testSwitchErrors();
  testJumpsErrors();
  testClassAccessErrors();
  testGotoErrors();
  testAttributeErrors();
  std::cerr << "==============================================\n\n";
}

void testExpressionErrors() {
  parseBadSource("2 + 3");
  parseBadSource("-2");
  parseBadSource("2 = {}");
  parseBadSource("sizeof(4)");
}

void testDeclarationErrors() {
  parseBadSource("int foo");
  parseBadSource("int foo = 3");
  parseBadSource("int foo = 3, bar = 4");
  parseBadSource("int foo = 3, *bar = 4");
}

void testNamespaceErrors() {
  parseBadSource("namespace foo");
  parseBadSource("namespace foo::");
  parseBadSource("namespace foo::bar::");
  parseBadSource("namespace foo + {}");
}

void testTypeDeclErrors() {
}

void testFunctionErrors() {
  parseBadSource("int foo()");
}

void testIfErrors() {
  parseBadSource("if (true)");
  parseBadSource("if () {}");
  parseBadSource("if (if (true) {}) {}");
  parseBadSource("if (;;) {}");

  parseBadSource("if (;) @attr {}");
}

void testForErrors() {
  // parseBadSource("for () {}");
  // parseBadSource("for (;) {}");
  // parseBadSource("for (;;)");
  // parseBadSource("for (;;;;) {}");

  parseBadSource("for (;;) @attr {}");
}

void testWhileErrors() {
  parseBadSource("while (;;) {}");
  parseBadSource("do {};");
  parseBadSource("do;");
  parseBadSource("do {} while (;;);");
  parseBadSource("do {} while (true)");
  parseBadSource("do ; while (true)");
  parseBadSource("do {} while (int i = 0)");

  parseBadSource("while (;) @attr {}");
  parseBadSource("do {} while (int i = 0) @attr;");
}

void testSwitchErrors() {
  parseBadSource("switch ()");
  parseBadSource("switch (true)");
  parseBadSource("switch (true) case 2:");
  parseBadSource("switch (true) default:");
  parseBadSource("switch (;;) {}");

  parseBadSource("switch (true) @attr {}");
  parseBadSource("switch (true) @attr default:;");
}

void testJumpsErrors() {
  parseBadSource("continue");
  parseBadSource("break");
  parseBadSource("return");
  parseBadSource("return 1 + 2");
}

void testClassAccessErrors() {
  parseBadSource("public");
  parseBadSource("protected");
  parseBadSource("private");
}

void testGotoErrors() {
  parseBadSource("goto");
  parseBadSource("goto;");
}

void testAttributeErrors() {
  parseBadSource("@attr");
  parseBadSource("@attr()");

  parseBadSource("@dim;");
  parseBadSource("@dim int x;");
  parseBadSource("@dim() int x;");
  parseBadSource("@dim(x=1) int x;");

  parseBadSource("@tile(16);");
  parseBadSource("@tile(16, x=1);");
  parseBadSource("@tile(16, check='o');");
  parseBadSource("for (i = 0;;; @tile(16)) {}");
  parseBadSource("for (float i = 0;;; @tile(16)) {}");
  parseBadSource("for (int i = 0, j = 0;;; @tile(16)) {}");
  parseBadSource("for (int i = 0;;; @tile(16)) {}");
  parseBadSource("for (int i = 0; i + 2;; @tile(16)) {}");
  parseBadSource("for (int i = 0; j < 2;; @tile(16)) {}");
  parseBadSource("for (int i = 0; i < 2;; @tile(16)) {}");
  parseBadSource("for (int i = 0; i < 2; i *= 2; @tile(16)) {}");
  parseBadSource("for (int i = 0; i < 2; ++j; @tile(16)) {}");

  parseBadSource("@dimOrder(1, 0);");
  parseBadSource("@dimOrder() int x;");
  parseBadSource("@dimOrder(,) int x;");
  parseBadSource("@dimOrder(1,x,0) int x;");
  parseBadSource("@dimOrder(0,1,2,4) int x;");
  parseBadSource("@dimOrder(-1,1,2,4) int x;");
  parseBadSource("@dimOrder(11) int x;");
}
//======================================

//---[ Scope ]--------------------------
const std::string scopeTestSource = (
  "int x;\n"
  "typedef int myInt;\n"
  "\n"
  "void foo() {\n"
  "  int x;\n"
  "  {\n"
  "    int x;\n"
  "  }\n"
  "  typedef int myInt;\n"
  "}\n"
  "\n"
  "int main(const int argc, const char **argv) {\n"
  "  int x = argc;\n"
  "  int a;\n"
  "  if (true) {\n"
  "    int x = 0;\n"
  "    int b;\n"
  "    if (true) {\n"
  "      int x = 1;\n"
  "      int c;\n"
  "      if (true) {\n"
  "        int x = 2;\n"
  "        int d;\n"
  "      }\n"
  "    }\n"
  "  }\n"
  "}\n");

void testScopeUp();
void testScopeKeywords();
void testScopeErrors();

void testScope() {
  testScopeUp();
  testScopeKeywords();

  testScopeErrors();
}

void testScopeUp() {
  parseSource(scopeTestSource);

  blockStatement &root = parser.root;

  statement_t *x           = root[0];
  blockStatement &foo      = root[2]->to<blockStatement>();
  blockStatement &main     = root[3]->to<blockStatement>();
  blockStatement &fooBlock = foo[1]->to<blockStatement>();

  ASSERT_EQ(&root,
            x->up);
  ASSERT_EQ(&root,
            foo.up);
  ASSERT_EQ(&root,
            main.up);
  ASSERT_EQ(&foo,
            fooBlock.up);
}

void testScopeKeywords() {
  parseSource(scopeTestSource);

  blockStatement &root     = parser.root;
  blockStatement &foo      = root[2]->to<blockStatement>();
  blockStatement &fooBlock = foo[1]->to<blockStatement>();

  // Make sure we can find variables 'x'
  ASSERT_TRUE(root.inScope("x"));
  ASSERT_TRUE(foo.inScope("x"));
  ASSERT_TRUE(fooBlock.inScope("x"));

  // Make sure variables 'x' exist
  ASSERT_EQ_BINARY(keywordType::variable,
                   root.getScopeKeyword("x").type());
  ASSERT_EQ_BINARY(keywordType::variable,
                   foo.getScopeKeyword("x").type());
  ASSERT_EQ_BINARY(keywordType::variable,
                   fooBlock.getScopeKeyword("x").type());

  // Make sure all instances are different
  ASSERT_NEQ(&root.getScopeKeyword("x").to<variableKeyword>().variable,
             &foo.getScopeKeyword("x").to<variableKeyword>().variable);

  ASSERT_NEQ(&root.getScopeKeyword("x").to<variableKeyword>().variable,
             &fooBlock.getScopeKeyword("x").to<variableKeyword>().variable);

  ASSERT_NEQ(&foo.getScopeKeyword("x").to<variableKeyword>().variable,
             &fooBlock.getScopeKeyword("x").to<variableKeyword>().variable);

  // Test function
  ASSERT_EQ_BINARY(keywordType::function,
                   root.getScopeKeyword("foo").type());
  ASSERT_EQ_BINARY(keywordType::function,
                   root.getScopeKeyword("main").type());

  // Test types
  ASSERT_EQ_BINARY(keywordType::type,
                   root.getScopeKeyword("myInt").type());
  ASSERT_EQ_BINARY(keywordType::type,
                   foo.getScopeKeyword("myInt").type());
}

void testScopeErrors() {
  std::cerr << "\n---[ Testing scope errors ]---------------------\n\n";
  const std::string var = "int x;\n";
  const std::string type = "typedef int x;\n";
  const std::string func = "void x() {}\n";
  std::string sources[3] = { var, type, func };

  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      parseBadSource(sources[j] + sources[i]);
      std::cout << '\n';
    }
  }

  parseBadSource("int x, x;\n");
  std::cerr << "==============================================\n\n";
}
//======================================
