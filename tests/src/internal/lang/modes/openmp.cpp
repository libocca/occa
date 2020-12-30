#define OCCA_TEST_PARSER_TYPE okl::openmpParser

#include <occa/internal/lang/modes/openmp.hpp>
#include "../parserUtils.hpp"

void testPragma();
void testAtomic();

int main(const int argc, const char **argv) {
  parser.settings["okl/validate"] = false;
  parser.settings["serial/include_std"] = false;

  testPragma();
  testAtomic();

  return 0;
}

#define ASSERT_PRAGMA_EXISTS(PRAGMA_SOURCE, COUNT)                      \
  do {                                                                  \
    statementArray pragmaStatements = (                                 \
      parser.root.children                                              \
      .flatFilterByStatementType(statementType::pragma)                 \
    );                                                                  \
                                                                        \
    ASSERT_EQ(COUNT,                                                    \
              (int) pragmaStatements.length());                         \
                                                                        \
    pragmaStatement &ompPragma = pragmaStatements[0]->to<pragmaStatement>(); \
    ASSERT_EQ(PRAGMA_SOURCE,                                            \
              ompPragma.value());                                       \
  } while(0)

//---[ Pragma ]-------------------------
void testPragma() {
  // @outer -> #pragma omp
  parseSource(
    "@kernel void foo() {\n"
    "  for (;;; @outer) {}\n"
    "}"
  );
  ASSERT_PRAGMA_EXISTS("omp parallel for", 1);
}
//======================================

//---[ @atomic ]------------------------
void testAtomic() {
  parseSource(
    "int i;\n"
    "@atomic i += 1;\n"
  );
  ASSERT_PRAGMA_EXISTS("omp atomic", 1);

  parseSource(
    "@atomic i < 1;\n"
  );
  ASSERT_PRAGMA_EXISTS("omp critical", 1);

  parseSource(
    "int i;\n"
    "@atomic {\n"
    "  i += 1;\n"
    "}\n"
  );
  ASSERT_PRAGMA_EXISTS("omp atomic", 1);

  parseSource(
    "int i;\n"
    "@atomic {\n"
    "  i += 1;\n"
    "  i += 1;\n"
    "}\n"
  );
  ASSERT_PRAGMA_EXISTS("omp critical", 1);
}
//======================================
