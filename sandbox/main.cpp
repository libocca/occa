#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    void test(){
      parser p;
      statement s(p);

      // strNode *n = labelCode( splitContent("#pragma blah") );
      strNode *n = labelCode( splitContent("const int *a = NULL, b= a, **c; int b;") );
      // strNode *n = labelCode( splitContent("const int * const * (*func)(int **x, int, int)") );
      // strNode *n = labelCode( splitContent("(1+2*3%2|1+10&3^1)") );

      expNode expRoot(s);
      expRoot.loadFromNode(n);

      std::cout
        << "expRoot = " << expRoot << '\n';

      expRoot.print();

      throw 1;
    }
  };
};

int main(int argc, char **argv){
  // occa::parserNamespace::test();

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/easy.c");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/test.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/openclTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/cudaTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/midg.okl");
  //   std::cout << parsedContent << '\n';
  // }

  {
    occa::parser parser;
    std::string parsedContent = parser.parseFile("tests/cleanTest.c");
    std::cout << parsedContent << '\n';
  }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/clangTest.c");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/addVectors.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/PCGpart1.cl");
  //   std::cout << parsedContent << '\n';
  // }
}
