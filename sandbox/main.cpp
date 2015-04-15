#include "occaParser.hpp"

using namespace occa;
using namespace occa::parserNS;

int main(int argc, char **argv){
  occa::parser parser;
  parser.warnForMissingBarriers     = false;
  parser.warnForBarrierConditionals = false;

  // typeHolder result = evaluateString("3.3*(1 + 2)");
  // std::cout << result << '\n';

  // strToStrMap_t stsMap;
  // stsMap["a"] = "2";
  // stsMap["b"] = "3";

  // parser.parseSource("int a, b, stride; a = a + b*stride;");
  // expNode &e = parser.globalScope->statementStart->right->value->expRoot;

  // // std::cout << e.valueIsKnown(stsMap) << '\n';
  // // std::cout << e.computeKnownValue(stsMap) << '\n';

  // // return 0;

  // accessInfo ai;
  // ai.load(e);
  // return 0;

  // {
  //   std::string parsedContent = parser.parseFile("tests/easy.c");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/test.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/scratch.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/scratch.ofl",
  //                                                occa::parserNS::parsingFortran);
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/pwdg.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/openclTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/cudaTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/fd2d_cuda.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/midg.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/cleanTest.c");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/clangTest.c");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/addVectors.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/addVectors.ofl",
  //                                                occa::parserNS::parsingFortran);
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/fakeNuma.ofl",
  //                                                occa::parserNS::parsingFortran);
  //   std::cout << parsedContent << '\n';
  // }

  {
    parser.magicEnabled = true;

    std::string parsedContent = parser.parseFile("tests/ridgV.oak");
    std::cout << parsedContent << '\n';
  }

  // {
  //   std::string parsedContent = parser.parseFile("tests/PCGpart1.cl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/lookup_kernel.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/reduction.cl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/loopy.cl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/addVectors_loopy.cl");
  //   std::cout << parsedContent << '\n';
  // }
}
