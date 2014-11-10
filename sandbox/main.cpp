#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    void test(){
      parser p;
      p.loadLanguageTypes();
      statement &s = *(p.globalScope);

      // strNode *nodeRoot = p.splitAndPreprocessContent("typedef struct a { int b, c; struct b {};} *b2;");
      strNode *nodeRoot = p.splitAndPreprocessContent("const int * func(const int a, const int &b){}");

      // strNode *nodeRoot = p.splitAndPreprocessContent("const int *const ** const***a[2], *b, ((c)), d[3], e(int), (f), ((*g))(), (*(*h)(int))(double), (*(*(*i)())(int))(double);");

#if 1
      const int varCount = varInfo::variablesInStatement(nodeRoot);

      if(varCount){
        varInfo *variables = new varInfo[varCount];

        nodeRoot = variables[0].loadFrom(s, nodeRoot);
        std::cout << "variables[0] = " << variables[0] << '\n';

        for(int i = 1; i < varCount; ++i){
          nodeRoot = variables[i].loadFrom(s, nodeRoot, &(variables[0]));
          std::cout << "variables[" << i << "] = " << variables[i] << '\n';
        }
      }
#else
      typeInfo type;
      nodeRoot = type.loadFrom(s, nodeRoot);
      std::cout << "type = " << type << '\n';
#endif

        // expNode *expRoot = addNewVariables(nodeRoot);
        // expRoot->print();

        throw 1;
    }
  };
};

int main(int argc, char **argv){
  occa::parserNamespace::test();

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

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/cleanTest.c");
  //   std::cout << parsedContent << '\n';
  // }

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

  {
    occa::parser parser;
    std::string parsedContent = parser.parseFile("tests/lookup_kernel.okl");
    std::cout << parsedContent << '\n';
  }
}
