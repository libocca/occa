#define OCCA_TEST_PARSER_TYPE okl::dpcppParser

#include <occa/internal/lang/modes/dpcpp.hpp>
#include "../parserUtils.hpp"

#undef parseAndPrintSource
#define parseAndPrintSource(str_)                                       \
  parseSource(str_);                                                    \
  ASSERT_TRUE(parser.success);                                          \
  {                                                                     \
    printer pout;                                                       \
    parser.launcherParser.root.print(pout);                                 \
    std::cout << "---[ Host ]-----------------------------------\n";    \
    std::cout << pout.str();                                            \
    std::cout << "==============================================\n\n";  \
    pout.clear();                                                       \
    parser.root.print(pout);                                            \
    std::cout << "---[ Device ]---------------------------------\n";    \
    std::cout << pout.str();                                            \
    std::cout << "==============================================\n\n";  \
  }

void testLoopExtraction();
void testGlobalConst();
void testKernelAnnotation();
void testKernelArgs();
void testSharedAnnotation();
void testBarriers();
void testAtomic();
void testSource();

int main(const int argc, const char **argv) {
  parser.settings["okl/validate"] = true;
  testLoopExtraction();
  testGlobalConst();
  testKernelAnnotation();
  testKernelArgs();
  testSharedAnnotation();
  testBarriers();
  testSource();

  return 0;
}

//---[ Loops ]--------------------------
void testLoopExtraction() {
}
//======================================

//---[ Constant ]-----------------------
void testGlobalConst() {
}
//======================================

//---[ Kernel ]-------------------------
void testKernelAnnotation() {
}
//======================================

//---[ Kernel Args ]--------------------
void testKernelArgs() {
}
//======================================

//---[ Shared ]-------------------------
void testSharedAnnotation() {
}
//======================================

//---[ @atomic ]------------------------
void testAtomic() {
}
//======================================

//---[ Barriers ]-----------------------
void testBarriers() {
}
//======================================

void testSource() {
  
  parseAndPrintSource(
    "const int var[10];\n"
    "void foo() {}\n"
    "int bar(int i) {}\n"
    "@kernel void kernel(@restrict int * arg, const int bar) {\n"
    "  for (int o1 = 0; o1 < O1; ++o1; @outer) {\n"
    "    for (int o0 = 0; o0 < O0; ++o0; @outer) {\n"
    "      @shared int shr[3];\n"
    "      @exclusive int excl;\n"
    "      if (true) {\n"
    "        for (int i1 = 10; i1 < (I1 + 4); i1 += 3; @inner) {\n"
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
    "  for (int o1 = 0; o1 < O1; ++o1; @outer(0)) {\n"
    "    for (int o0 = 0; o0 < O0; ++o0; @outer(1)) {\n"
    "      @shared int shr[3];\n"
    "      @exclusive int excl;\n"
    "      if (true) {\n"
    "        for (int i1 = 10; i1 < (I1 + 4); i1 += 3; @inner(1)) {\n"
    "          for (int i0 = 0; i0 < I0; ++i0; @inner(0)) {\n"
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
    "  for (int ib = 0; ib < entries; ib += 16; @outer) {\n"
    "    for (int it = 0; it < 16; ++it; @inner) {\n"
    "      const int i = ib + it;\n"
    "      if (i < entries) {\n"
    "        ab[i] = a[i] + b[i];\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
  parseAndPrintSource(
    "@kernel void addVectors(const int entries,\n"
    "                        const float *a,\n"
    "                        const float *b,\n"
    "                        float *ab) {\n"
    "  for (int i = 0; i < entries; ++i; @tile(16, @outer, @inner)) {\n"
    "    ab[i] = a[i] + b[i];\n"
    "  }\n"
    "}\n"
  );
  parseAndPrintSource(
    "@kernel void addVectors(const int entries,\n"
    "                        const float *a,\n"
    "                        const float *b,\n"
    "                        float *ab) {\n"
    "  for (int ib = 0; ib < entries; ib += 16; @outer) {\n"
    "    @shared int foo[10];\n"
    "    for (int it = 0; it < 16; ++it; @inner) {\n"
    "      const int i = ib + it;\n"
    "      if (i < entries) {\n"
    "        foo[i] = a[i] + b[i];\n"
    "      }\n"
    "    }\n"
    "    for (int it = 0; it < 16; ++it; @inner) {\n"
    "      const int i = ib + it;\n"
    "      if (i < entries) {\n"
    "        foo[i] = a[i] + b[i];\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}
