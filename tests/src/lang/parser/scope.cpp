#include "utils.hpp"

void testScopeUp();
void testScopeKeywords();
void testScopeErrors();

int main(const int argc, const char **argv) {
  setupParser();

  testScopeUp();
  testScopeKeywords();
  testScopeErrors();

  return 0;
}

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
  "}\n"
);

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
  ASSERT_TRUE(root.hasInScope("x"));
  ASSERT_TRUE(foo.hasInScope("x"));
  ASSERT_TRUE(fooBlock.hasInScope("x"));

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
