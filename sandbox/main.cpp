#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    namespace expType {
      const char L         = (1 << 0);
      const char C         = (1 << 1);
      const char R         = (1 << 2);
      const char type      = (1 << 3);
      const char var       = (1 << 4);
      const char function  = (1 << 5);
      const char pFunction = (1 << 6);
    };

    class expNode {
    public:
      std::string value;
      char info;

      int leafCount;
      expNode **leaves;
      varInfo *var;
      typeDef *type;

      void loadFromNode(strNode *n){
        strNode *nc = n->clone();
        nc->flatten();
      }
    };

    void test(){
      strNode *n = labelCode( splitContent("(1+2*3%2|1+10&3^1)") );

      expNode expRoot;
      expRoot.loadFromNode(n);

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
