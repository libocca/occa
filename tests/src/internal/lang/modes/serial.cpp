#define OCCA_TEST_PARSER_TYPE okl::serialParser

#include <occa/internal/lang/modes/serial.hpp>
#include "../parserUtils.hpp"

void testPreprocessor();
void testKernel();
void testExclusives();
void testAtomic();

int main(const int argc, const char **argv) {
  parser.settings["serial/include_std"] = false;

  // parser.settings["okl/validate"] = false;
  // testPreprocessor();
  // testKernel();

  // parser.settings["okl/validate"] = true;
  // testExclusives();

  return 0;
}

//---[ Preprocessor ]-------------------
void testPreprocessor() {
  // @restrict -> __restrict__
  statement_t *statement = NULL;

  parseAndPrintSource("@kernel void foo(@restrict const int * a) {}");
  setStatement("@kernel void foo(@restrict const int * a) {}",
               statementType::functionDecl);

  delete statement;
}
//======================================

//---[ @kernel ]------------------------
void testExtern();
void testArgs();

void testKernel() {
  testExtern();
  testArgs();
}

void testExtern() {
  // @kernel -> extern "C"

#define func (statement->to<functionDeclStatement>().function())

  statement_t *statement;
  setStatement("@kernel void foo() {}",
               statementType::functionDecl);

  ASSERT_TRUE(func.returnType.has(externC));

#undef func
}

void testArgs() {
  // @kernel args -> by reference
#define func       (statement->to<functionDeclStatement>().function())
#define arg(N)     (*(args[N]))
#define argType(N) (arg(N).vartype)

  statement_t *statement;
  setStatement("@kernel void foo(\n"
               "const int A,\n"
               "const int *B,\n"
               "const int &C,\n"
               ") {}",
               statementType::functionDecl);

  variablePtrVector &args = func.args;
  const int argCount = (int) args.size();
  ASSERT_EQ(3,
            argCount);

  ASSERT_EQ("A",
            arg(0).name());
  ASSERT_NEQ((void*) NULL,
             argType(0).referenceToken);

  ASSERT_EQ("B",
            arg(1).name());
  ASSERT_EQ((void*) NULL,
            argType(1).referenceToken);

  ASSERT_EQ("C",
            arg(2).name());
  ASSERT_NEQ((void*) NULL,
             argType(2).referenceToken);

#undef func
#undef arg
}
//======================================

//---[ @exclusive ]---------------------
void testExclusives() {
  // TODO:
  //   @exclusive ->
  //     - std::vector<value>
  //     - vec.reserve(loopIterations)
  //     - Add iterator index to inner-most @inner loop
  parseAndPrintSource(
    "const int var[10];\n"
    "@kernel void foo(@restrict int * arg) {\n"
    "  for (int o1 = 0; o1 < O1; ++o1; @outer) {\n"
    "    for (int o0 = 0; o0 < O0; ++o0; @outer) {\n"
    "      @shared int shr[3];\n"
    "      @exclusive int excl;\n"
    "      if (true) {\n"
    "        for (int i1 = 0; i1 < I1; ++i1; @inner) {\n"
    "          for (int i0 = 0; i0 < I0; ++i0; @inner) {\n"
    "            for (;;) {\n"
    "               excl = i0;\n"
    "            }\n"
    "            for (;;) {\n"
    "               excl = i0;\n"
    "            }\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}
//======================================

//---[ @atomic ]------------------------
void testAtomic() {
  // TODO(dmed)
}
//======================================
