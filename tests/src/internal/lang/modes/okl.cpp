#include "../parserUtils.hpp"
#include <occa/internal/lang/modes/okl.hpp>


#define parseOKLSource(src_)                    \
  parseSource(src_);                            \
  ASSERT_TRUE(okl::kernelsAreValid(parser.root))

#define parseBadOKLSource(src_)                 \
  parseSource(src_);                            \
  ASSERT_FALSE(okl::kernelsAreValid(parser.root))


void testKernel();
void testLoops();
void testTypes();
void testLoopSkips();

int main(const int argc, const char **argv) {
  parser.addAttribute<dummy>();
  parser.addAttribute<attributes::kernel>();
  parser.addAttribute<attributes::outer>();
  parser.addAttribute<attributes::inner>();
  parser.addAttribute<attributes::shared>();
  parser.addAttribute<attributes::exclusive>();
  parser.addAttribute<attributes::maxInnerDims>();

  testKernel();
  testLoops();
  testTypes();
  testLoopSkips();

  return 0;
}

void testKernel() {
  parseBadOKLSource("@kernel int foo() {}");
}

//---[ Loop ]---------------------------
void testOKLLoopExists();
void testProperOKLLoops();
void testInnerInsideOuter();
void testSameInnerLoopCount();
void testEmptyLoops();
void testInfiniteLoops();
void testMaxInnerDims();

void testLoops() {
  testOKLLoopExists();
  testProperOKLLoops();
  testInnerInsideOuter();
  testSameInnerLoopCount();
  testEmptyLoops();
  testInfiniteLoops();
  testMaxInnerDims();
}

void testOKLLoopExists() {
  // @outer + @inner exist
  parseBadOKLSource("@kernel void foo() {}");
  parseBadOKLSource("@kernel void foo() {\n"
                    "  for (;;; @outer) {}\n"
                    "}");
  parseBadOKLSource("@kernel void foo() {\n"
                    "  for (;;; @inner) {}\n"
                    "}");
}

void testProperOKLLoops() {
  // Proper loops (decl, update, inc)
  const std::string oStart = (
    "@kernel void foo() {\n"
  );
  const std::string oMid = (
    "\nfor (int i = 0; i < 2; ++i; @inner) {}\n"
  );
  const std::string oEnd = (
    "\n}\n"
  );

  const std::string iStart = (
    "@kernel void foo() {\n"
    "for (int o = 0; o < 2; ++o; @outer) {\n"
  );
  const std::string iEnd = (
    "\n}\n"
    "}\n"
  );

  parseBadOKLSource(oStart + "for (o = 0;;; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (float o = 0;;; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (int o = 0, j = 0;;; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (int o = 0;;; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (int o = 0; o + 2;; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (int o = 0; j < 2;; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (int o = 0; o < 2;; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (int o = 0; o < 2; o *= 2; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (int o = 0; o < 2; ++j; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (int o; o < 2; ++o; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for (int; o < 2; ++o; @outer) {" + oMid + "}" + oEnd);
  parseBadOKLSource(oStart + "for ( ; o < 2; ++o; @outer) {" + oMid + "}" + oEnd);

  parseBadOKLSource(iStart + "for (i = 0;;; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (float i = 0;;; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (int i = 0, j = 0;;; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (int i = 0;;; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (int i = 0; i + 2;; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (int i = 0; j < 2;; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (int i = 0; i < 2;; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (int i = 0; i < 2; i *= 2; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (int i = 0; i < 2; ++j; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (int i; i < 2; ++i; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for (int; i < 2; ++i; @inner) {}" + iEnd);
  parseBadOKLSource(iStart + "for ( ; i < 2; ++i; @inner) {}" + iEnd);

  // No double @outer + @inner
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int i = 0; i < 2; ++i; @outer @inner) {\n"
    "  }\n"
    "}\n"
  );
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int i = 0; i < 2; ++i; @tile(1, @outer @inner)) {\n"
    "  }\n"
    "}\n"
  );
  // Make sure @tile distributes the attributes properly
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int i = 0; i < 2; ++i; @tile(1, @inner, @outer)) {\n"
    "  }\n"
    "}\n"
  );
  parseOKLSource(
    "@kernel void foo() {\n"
    "  for (int i = 0; i < 2; ++i; @tile(1, @outer, @inner)) {\n"
    "  }\n"
    "}\n"
  );
}

void testInnerInsideOuter() {
  // @outer > @inner
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int i = 0; i < 2; ++i; @inner) {\n"
    "    for (int o = 0; o < 2; ++o; @outer) {}\n"
    "  }\n"
    "}\n"
  );
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "      for (int o2 = 0; o2 < 2; ++o2; @outer) {}\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}

void testSameInnerLoopCount() {
  // Same # of @inner in each @outer
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {}\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "      for (int i2 = 0; i2 < 2; ++i2; @inner) {\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}

void testEmptyLoops() {

  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 10; i < 2; i+=3; @inner) {\n"
    "        int k = i + j;\n"
    "    }\n"
    "  }\n"
    "}\n"
  );

  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 10; i > 11; i-=3; @inner) {\n"
    "        int k = i + j;\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}

void testInfiniteLoops() {

  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; i-=5; @inner) {\n"
    "        int k = i + j;\n"
    "    }\n"
    "  }\n"
    "}\n"
  );

  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 2; i > 2=0; i+=5; @inner) {\n"
    "        int k = i + j;\n"
    "    }\n"
    "  }\n"
    "}\n"
  );

}

void testMaxInnerDims() {
  parseBadOKLSource(
    "@kernel void foo(const int& N) {\n"
    "  @max_inner_dims(0)\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < N; ++i; @inner) {\n"
    "        int k = i + j;\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}
//======================================

//---[ Types ]--------------------------
void testSharedLocation();
void testExclusiveLocation();
void testValidSharedArray();

void testTypes() {
  testSharedLocation();
  testExclusiveLocation();
  testValidSharedArray();
}

void testSharedLocation() {
  // @outer > @shared > @inner
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  @shared int s[10];\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "      @shared int s[10];\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}

void testExclusiveLocation() {
  // @outer > @exclusive > @inner
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  @exclusive int x;\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "      @exclusive int x;\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}

void testValidSharedArray() {
  // @shared has an array with evaluable sizes
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    @shared int s[o];\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    @shared int s[2][o];\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    @shared int s[2][];\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}
//======================================

//---[ Loop Skips ]---------------------
void testValidBreaks();
void testValidContinues();

void testLoopSkips() {
  testValidBreaks();
  testValidContinues();
}

void testValidBreaks() {
  // No break in @outer/@inner (ok inside regular loops inside @outer/@inner)
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "    }\n"
    "    break;"
    "  }\n"
    "}\n"
  );
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "      break;"
    "    }\n"
    "  }\n"
    "}\n"
  );
}

void testValidContinues() {
  // No continue in @inner (ok inside regular loops inside @outer/@inner)
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "    }\n"
    "    continue;"
    "  }\n"
    "}\n"
  );
  parseBadOKLSource(
    "@kernel void foo() {\n"
    "  for (int o = 0; o < 2; ++o; @outer) {\n"
    "    for (int i = 0; i < 2; ++i; @inner) {\n"
    "      continue;"
    "    }\n"
    "  }\n"
    "}\n"
  );
}
//======================================
