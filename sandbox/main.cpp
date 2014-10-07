#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    class expNode {
      std::string value;
      char info;

      int leafCount;
      expNode **leaves;
      varInfo *var;
      typeDef *type;
    };

    void test(){
      strNode *n = labelCode( splitContent("1 + ((2 + 3) + 4)") );
      n->print();
      printf("|||||||||||||||||||||||||\n");
      n->flatten();
      n->print();

      throw 1;
    }
  };
};

int main(int argc, char **argv){
  occa::parserNamespace::test();

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
