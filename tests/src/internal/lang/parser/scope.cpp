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
  // root[0]
  "int x;\n"
  // root[1]
  "typedef int myInt;\n"
  // root[2]
  "void foo() {\n"
  "  int x;\n"
  "  {\n"
  "    int x;\n"
  "  }\n"
  "  typedef int myInt;\n"
  "}\n"
  // root[3]
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
  // root[4]
  "struct struct1_t {\n"
  "  int x1;\n"
  "};\n"
  // root[5]
  "typedef struct {\n"
  "  int x2;\n"
  "} struct2_t;\n"
  // root[6]
  "typedef struct struct3_t {\n"
  "  int x3, x4;\n"
  "} struct4_t;\n"
  // root[7]
  "struct struct1_t struct1;\n"
  // root[8]
  "struct struct2_t struct2;\n"
  // root[9]
  "struct struct3_t struct3;\n"
  // root[10]
  "struct struct4_t struct4;\n"
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
  ASSERT_TRUE(root[7]->is<declarationStatement>());
  ASSERT_TRUE(root[8]->is<declarationStatement>());
  ASSERT_TRUE(root[9]->is<declarationStatement>());
  ASSERT_TRUE(root[10]->is<declarationStatement>());

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

  // Test structs
  ASSERT_EQ_BINARY(keywordType::type,
                   root.getScopeKeyword("struct1_t").type());
  ASSERT_EQ_BINARY(keywordType::type,
                   root.getScopeKeyword("struct2_t").type());
  ASSERT_EQ_BINARY(keywordType::type,
                   root.getScopeKeyword("struct3_t").type());
  ASSERT_EQ_BINARY(keywordType::type,
                   root.getScopeKeyword("struct4_t").type());
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
