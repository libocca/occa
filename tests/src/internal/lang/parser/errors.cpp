#include "utils.hpp"

void testExpressionErrors();
void testDeclarationErrors();
void testNamespaceErrors();
void testFunctionErrors();
void testStructErrors();
void testIfErrors();
void testForErrors();
void testWhileErrors();
void testSwitchErrors();
void testJumpsErrors();
void testClassAccessErrors();
void testAttributeErrors();
void testGotoErrors();

void testErrors() {
}

int main(const int argc, const char **argv) {
  setupParser();

  std::cerr << "\n---[ Testing parser errors ]--------------------\n\n";
  testExpressionErrors();
  testDeclarationErrors();
  testNamespaceErrors();
  testFunctionErrors();
  testStructErrors();
  testIfErrors();
  testForErrors();
  testWhileErrors();
  testSwitchErrors();
  testJumpsErrors();
  testClassAccessErrors();
  testGotoErrors();
  testAttributeErrors();
  std::cerr << "==============================================\n\n";

  return 0;
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

void testFunctionErrors() {
  parseBadSource("int foo()");
}

void testStructErrors() {
  parseBadSource("struct {}");
  parseBadSource("struct foo {}");
  parseBadSource("struct foo;");
  parseBadSource("struct foo {\n"
                 "  3;\n"
                 "};");
  parseBadSource("struct foo {\n"
                 "  void bar();\n"
                 "};");
  parseBadSource("struct foo {\n"
                 "  public:\n"
                 "  int x, y, z;\n"
                 "};");
  parseBadSource("struct foo {\n"
                 "  protected:\n"
                 "  int x, y, z;\n"
                 "};");
  parseBadSource("struct foo {\n"
                 "  private:\n"
                 "  int x, y, z;\n"
                 "};");
  parseBadSource("struct foo {\n"
                 "  int x = 3;\n"
                 "};");
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
