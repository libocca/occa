#define OCCA_TEST_PARSER_TYPE okl::openmpParser

#include <occa/lang/mode/openmp.hpp>
#include "../parserUtils.hpp"

void testPragma();

int main(const int argc, const char **argv) {
  parser.settings["okl/validate"] = false;
  parser.settings["serial/include-std"] = false;

  // testPragma();

  return 0;
}

//---[ Pragma ]-------------------------
void testPragma() {
  // @outer -> #pragma omp
  parseSource(
    "@kernel void foo() {\n"
    "  for (;;; @outer) {}\n"
    "}"
  );

  ASSERT_EQ(1,
            parser.root.size());

  functionDeclStatement &foo = parser.root[0]->to<functionDeclStatement>();
  ASSERT_EQ(2,
            foo.size());

  pragmaStatement &ompPragma = foo[0]->to<pragmaStatement>();
  ASSERT_EQ("omp parallel for",
            ompPragma.value());
}
//======================================
