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

    int typeInfo::loadFrom(expNode &expRoot,
                           int leafPos){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      leafPos = leftQualifiers.loadFrom(expRoot, leafPos);

      if(leftQualifiers.has("typedef"))
        return loadTypedefFrom(expRoot, leafPos);

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].info & expType::unknown)){

        name = expRoot[leafPos++].value;
      }

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "{")){

        expNode &leaf = expRoot[leafPos++];

        const bool usesSemicolon = !leftQualifiers.has("enum");
        const char *delimiter = (usesSemicolon ? ";" : ",");

        nestedInfoCount = statementCountWithDelimeter(leaf, delimiter);
        nestedExps      = new expNode[nestedInfoCount];

        int sLeafPos = 0;

        for(int i = 0; i < nestedInfoCount; ++i){
          int sNextLeafPos = nextDelimeter(leaf, sLeafPos, delimiter);

          // Empty statements
          if(sNextLeafPos != sLeafPos){
            sNextLeafPos = leaf.mergeRange(expType::root,
                                           sLeafPos,
                                           sNextLeafPos);

            expNode::swap(nestedExps[i], leaf[sLeafPos]);

            nestedExps[i].print();
            nestedExps[i].organize();
            nestedExps[i].print();

            leaf.leaves[sLeafPos] = &(nestedExps[i]);
          }
          else{
            --i;
            --nestedInfoCount;
            ++sNextLeafPos;
          }

          sLeafPos = sNextLeafPos;
        }
      }

      return leafPos;
    }

    int typeInfo::statementCountWithDelimeter(expNode &expRoot,
                                              const char *delimiter){
      int count = 0;

      for(int i = 0; i < expRoot.leafCount; ++i){
        if(expRoot[i].value == delimiter)
          ++count;
      }

      return count;
    }

    int typeInfo::nextDelimeter(expNode &expRoot,
                                int leafPos,
                                const char *delimiter){
      for(int i = leafPos; i < expRoot.leafCount; ++i){
        if(expRoot[i].value == delimiter)
          return i;
      }

      return expRoot.leafCount;
    }

    int typeInfo::loadTypedefFrom(expNode &expRoot,
                                  int leafPos){
      return leafPos;
#if 0
      qualifierInfo newQuals = leftQualifiers.clone();
      newQuals.remove("typedef");

      leftQualifiers.remove(1, (leftQualifiers.qualifierCount - 1));

      if(nodePos->type != startBrace){
        typeInfo *tmp = s.hasTypeInScope(nodePos->value);

        if(tmp){
          typedefing = tmp;
        }
        else{
          typedefing           = new typeInfo;
          typedefing->name     = nodePos->value;
          typedefing->baseType = typedefing;
        }

        nodePos = nodePos->right;
      }

      if(nodePos->type == startBrace){
        if(typedefing == NULL){
          typedefing           = new typeInfo;
          typedefing->baseType = typedefing;
        }

        nodePos = typedefing->loadFrom(s, nodePos);
      }

      baseType = typedefing->baseType;

      varInfo typedefVarInfo;
      typedefVarInfo.baseType = typedefing;

      typedefVar = new varInfo;
      nodePos = typedefVar->loadFrom(s, nodePos, &typedefVarInfo);

      name = typedefVar->name;

      return nodePos;
#endif
    }

    void test2(){
      parser p;
      p.loadLanguageTypes();
      statement &s = *(p.globalScope);
      strNode *nodeRoot = p.splitAndPreprocessContent("struct a {int b;int c; double d,e; struct f {float g;};};");
      // strNode *nodeRoot = p.splitAndPreprocessContent("const int * const func(const int a, const int &b){}");
      strNode *nodePos = nodeRoot;

      s.expRoot.labelStatement(nodePos);

      strNode *newNodeRoot = nodeRoot->cloneTo(nodePos);

      s.expRoot.initLoadFromNode(newNodeRoot);

      s.expRoot.print();

      typeInfo type;
      int leafPos = type.loadFrom(s.expRoot, 0);

      std::cout << "type = " << type << '\n';

      throw 1;
    }
  };
};

int main(int argc, char **argv){
  // occa::parserNamespace::test();
  occa::parserNamespace::test2();

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
