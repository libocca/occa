#include "utils.hpp"
#include <occa/internal/utils/misc.hpp>

void testExpressionLoading();
void testDeclarationLoading();
void testNamespaceLoading();
void testStructLoading();
void testClassLoading();
void testUnionLoading();
void testEnumLoading();
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

int main(const int argc, const char **argv) {
  setupParser();

  testAttributeLoading();

  testExpressionLoading();
  testDeclarationLoading();
  testNamespaceLoading();
  testStructLoading();
  // testClassLoading();
  // testUnionLoading();
  // testEnumLoading();
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

  return 0;
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
#define declVar(N)   (decls[N].variable())
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

void testStructLoading() {
  statement_t *statement = NULL;
  struct_t *structType = NULL;
  typedef_t *typedefType = NULL;

#define declSmnt         statement->to<declarationStatement>()
#define getDeclType      declSmnt.declarations[0].variable().vartype.type
#define setStructType()  structType = (struct_t*) getDeclType
#define setTypedefType() typedefType = (typedef_t*) getDeclType

  // Test default struct
  setStatement(
    "struct vec3 {\n"
    "  int x, *y, &z;\n"
    "};",
    statementType::declaration
  );

  setStructType();

  ASSERT_EQ("vec3",
            structType->name());

  ASSERT_EQ(3,
            (int) structType->fields.size());

  ASSERT_EQ("x",
            structType->fields[0].name());
  ASSERT_EQ(&int_,
            structType->fields[0].vartype.type);

  ASSERT_EQ("y",
            structType->fields[1].name());
  ASSERT_EQ(&int_,
            structType->fields[1].vartype.type);

  ASSERT_EQ("z",
            structType->fields[2].name());
  ASSERT_EQ(&int_,
            structType->fields[2].vartype.type);

  // Test default typedef struct
  setStatement(
    "typedef struct vec3_t {\n"
    "  int x, *y, &z;\n"
    "} vec3;",
    statementType::declaration
  );

  setTypedefType();

  ASSERT_EQ("vec3",
            typedefType->name());

  ASSERT_EQ("vec3_t",
            typedefType->baseType.name());

  // Test typedef anonymous struct
  setStatement(
    "typedef struct {\n"
    "  int x, *y, &z;\n"
    "} vec3;",
    statementType::declaration
  );

  setTypedefType();

  ASSERT_EQ("vec3",
            typedefType->name());

  ASSERT_EQ(0,
            (int) typedefType->baseType.name().size());

#undef declSmnt
#undef getDeclType
#undef getStructType
#undef getTypedefType
}

void testClassLoading() {
  // TODO: Add class tests
}

void testUnionLoading() {
  // TODO: Add union tests
}

void testEnumLoading() {
  // TODO: Add enum tests

}

void testFunctionLoading() {
  statement_t *statement;

#define funcSmnt     statement->to<functionStatement>()
#define funcDeclSmnt statement->to<functionDeclStatement>()
#define func         funcSmnt.function()
#define funcDecl     funcDeclSmnt.function()

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
 
            funcDecl.name()) ; 
 
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
#define declVar(N)   (decls[N].variable())
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
            (int) forSmnt.children.length());

  setStatement("for (;;);",
               statementType::for_);
  ASSERT_EQ_BINARY(statementType::empty,
                   init.type());
  ASSERT_EQ_BINARY(statementType::empty,
                   check.type());
  ASSERT_EQ_BINARY(statementType::empty,
                   update.type());
  ASSERT_EQ(1,
            (int) forSmnt.children.length());

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
  occa::ignoreResult(statement);
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
            switchSmnt.size());

    // Weird cases
    setStatement("switch (2) ;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(1,
            switchSmnt.size());

    // Weird cases
    setStatement("switch (2) case 2:;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(2,
            switchSmnt.size());

    setStatement("switch (2) default:;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(2,
            switchSmnt.size());

    setStatement("switch (2) case 2: 2;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(2,
            switchSmnt.size());

    setStatement("switch (2) default: 2;",
                 statementType::switch_);
  ASSERT_EQ_BINARY(statementType::expression,
                   condition.type());
  ASSERT_EQ(2,
            switchSmnt.size());

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

  occa::ignoreResult(statement);
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
#define declVar(N)        (decls[N].variable())
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

  std::cerr << "\n---[ @restrict Transformations ]------------\n";
  parseAndPrintSource("void foo(@restrict int *a) {}");
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
