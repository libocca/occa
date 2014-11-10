#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    _qualifierInfo::_qualifierInfo() :
      qualifierCount(0),
      qualifiers(NULL) {}

    _qualifierInfo::_qualifierInfo(const _qualifierInfo &q) :
      qualifierCount(q.qualifierCount),
      qualifiers(q.qualifiers) {}

    _qualifierInfo& _qualifierInfo::operator = (const _qualifierInfo &q){
      qualifierCount = q.qualifierCount;
      qualifiers     = q.qualifiers;

      return *this;
    }

    _qualifierInfo _qualifierInfo::clone(){
      _qualifierInfo q;

      q.qualifierCount = qualifierCount;

      q.qualifiers = new std::string[qualifierCount];

      for(int i = 0; i < qualifierCount; ++i)
        q.qualifiers[i] = qualifiers[i];

      return q;
    }

    strNode* _qualifierInfo::loadFrom(statement &s,
                                      strNode *nodePos){
      strNode *nodeRoot = nodePos;

      while(nodePos &&
            s.nodeHasQualifier(nodePos)){
        ++qualifierCount;
        nodePos = nodePos->right;
      }

      if(qualifierCount){
        qualifiers = new std::string[qualifierCount];
        nodePos = nodeRoot;

        for(int i = 0; i < qualifierCount; ++i){
          qualifiers[i] = nodePos->value;
          nodePos = nodePos->right;
        }
      }

      return nodePos;
    }

    std::string _qualifierInfo::toString(){
      std::string ret;

      for(int i = 0; i < qualifierCount; ++i){
        ret += qualifiers[i];

        if((qualifiers[i][0] != '*') ||
           ( ((i + 1) < qualifierCount) &&
             (qualifiers[i + 1][0] != '*') )){

          ret += ' ';
        }
      }

      return ret;
    }

    _varInfo::_varInfo() :
      info(0),

      leftQualifiers(),
      rightQualifiers(),

      baseType(NULL),

      name(""),

      pointerCount(0),
      stackPointerCount(0),
      stackExpRoots(NULL),

      argumentCount(0),
      argumentVarInfos(NULL),

      nestedFunctionPointer(NULL) {}

    _varInfo::_varInfo(const _varInfo &var) :
      info(var.info),

      leftQualifiers(var.leftQualifiers),
      rightQualifiers(var.rightQualifiers),

      baseType(var.baseType),

      name(var.name),

      pointerCount(var.pointerCount),
      stackPointerCount(var.stackPointerCount),
      stackExpRoots(var.stackExpRoots),

      argumentCount(var.argumentCount),
      argumentVarInfos(var.argumentVarInfos),

      nestedFunctionPointer(var.nestedFunctionPointer) {}

    _varInfo& _varInfo::operator = (const _varInfo &var){
      info = var.info;

      leftQualifiers  = var.leftQualifiers;
      rightQualifiers = var.rightQualifiers;

      baseType = var.baseType;

      name = var.name;

      pointerCount      = var.pointerCount;
      stackPointerCount = var.stackPointerCount;
      stackExpRoots     = var.stackExpRoots;

      argumentCount    = var.argumentCount;
      argumentVarInfos = var.argumentVarInfos;

      nestedFunctionPointer = var.nestedFunctionPointer;

      return *this;
    }

    int _varInfo::variablesInStatement(strNode *nodePos){
      int argc = 0;

      while(nodePos){
        if((nodePos->value == ",") ||
           (nodePos->value == ";")){

          ++argc;
        }

        nodePos = nodePos->right;
      }

      return argc;
    }

    strNode* _varInfo::loadValueFrom(statement &s,
                                     strNode *nodePos,
                                     _varInfo *varHasType){
      nodePos = loadTypeFrom(s, nodePos, varHasType);

      info = getVarInfoFrom(s, nodePos);
      std::cout << "var.info = " << info << '\n';

      nodePos = loadNameFrom(s, nodePos);
      nodePos = loadArgsFrom(s, nodePos);

      if(nodePos->value == ",")
        nodePos = nodePos->right;

      return nodePos;
    }

    strNode* _varInfo::loadTypeFrom(statement &s,
                                    strNode *nodePos,
                                    _varInfo *varHasType){
      if(varHasType == NULL){
        nodePos = leftQualifiers.loadFrom(s, nodePos);

        baseType = s.hasTypeInScope(nodePos->value);

        if(baseType)
          nodePos = nodePos->right;
      }
      else{
        leftQualifiers = varHasType->leftQualifiers;
        baseType       = varHasType->baseType;
      }

      nodePos = rightQualifiers.loadFrom(s, nodePos);

      return nodePos;
    }

    int _varInfo::getVarInfoFrom(statement &s,
                                 strNode *nodePos){
      if(nodePos == NULL)
        return 0;

      strNode *nextNode = nodePos->right;

      while(nodePos &&
            (nodePos->type == startParentheses)){

        nodePos = nodePos->down;

        if(nodePos &&
           nodePos->value == "*"){

          return _varType::functionPointer;
        }
      }

      std::cout << "nextNode = " << *nextNode << '\n';

      if(nextNode &&
         (nextNode->type == startParentheses))
        return _varType::function;

      return _varType::var;
    }

    strNode* _varInfo::loadNameFrom(statement &s,
                                    strNode *nodePos){
      if(nodePos == NULL)
        return NULL;

      strNode *nextNode = nodePos->right;

      while(nodePos &&
            (nodePos->type == startParentheses)){

        nodePos = nodePos->down;

        if(nodePos &&
           nodePos->value == "*")
          nodePos = nodePos->right;
      }

      if(nodePos &&
         (nodePos->type & unknownVariable)){

        name    = nodePos->value;
        nodePos = nodePos->right;

        if(nodePos == nextNode)
          nextNode = loadStackPointersFrom(s, nextNode);
        else
          nodePos = loadStackPointersFrom(s, nodePos);
      }

      return nextNode;
    }

    strNode* _varInfo::loadStackPointersFrom(statement &s,
                                             strNode *nodePos){
      strNode *nodeRoot = nodePos;

      if(nodePos &&
         (nodePos->value == "[") &&
         (nodePos->down)){

        ++stackPointerCount;
        nodePos = nodePos->right;
      }

      if(stackPointerCount){
        nodePos = nodeRoot;

        stackExpRoots = new expNode[stackPointerCount];

        for(int i = 0; i < stackPointerCount; ++i){
          stackExpRoots[i].sInfo = &s;

          if(nodePos->down){
            strNode *downNode = nodePos->down;
            strNode *lastDown = lastNode(downNode);

            // s.setExpNodeFromStrNode(stackExpRoots[i], downNode);
          }

          nodePos = nodePos->right;
        }
      }

      return nodePos;
    }

    strNode* _varInfo::loadArgsFrom(statement &s,
                                    strNode *nodePos){
      if( !(info & _varType::functionType) )
        return nodePos;

      if(nodePos == NULL){
        std::cout << "Missing arguments from function variable\n";
        throw 1;
      }

      strNode *nextNode = nodePos->right;

      d

      return nextNode;
    }

    std::string _varInfo::getString(const bool printType){
      std::string ret;

      if(printType){
        ret += leftQualifiers.toString();
        ret += baseType->typeName;
        ret += ' ';
      }

      ret += rightQualifiers.toString();
      ret += name;

      for(int i = 0; i < stackPointerCount; ++i){
        ret += '[';
        ret += (std::string) stackExpRoots[i];
        ret += ']';
      }

      return ret;
    }

    _varInfo::operator std::string (){
      return getString();
    }

    std::ostream& operator << (std::ostream &out, _varInfo &var){
      out << var.getString();
      return out;
    }

    expNode* addNewVariables(strNode *nodePos){
      return NULL;
    }

    void test(){
      parser p;
      p.loadLanguageTypes();
      statement &s = *(p.globalScope);

      strNode *nodeRoot = p.splitAndPreprocessContent("const int *const ** const***a[2], *b, ((c)), d[3], e(int), (f), ((*g))();");

      const int varCount = _varInfo::variablesInStatement(nodeRoot);

      if(varCount){
        _varInfo *variables = new _varInfo[varCount];

        nodeRoot = variables[0].loadValueFrom(s, nodeRoot);
        std::cout << "variables[0] = " << variables[0] << '\n';

        for(int i = 1; i < varCount; ++i){
          nodeRoot = variables[i].loadValueFrom(s, nodeRoot, &(variables[0]));
          std::cout << "variables[" << i << "] = " << variables[i] << '\n';
        }
      }

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
