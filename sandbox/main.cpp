#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    //---[ Qualifier Info Class ]-----------------
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

    bool _qualifierInfo::has(const std::string &qName){
      for(int i = 0; i < qualifierCount; ++i){
        if(qualifiers[i] == qName)
          return true;
      }

      return false;
    }

    const std::string& _qualifierInfo::get(const int pos){
      if((pos < 0) ||
         (qualifierCount <= pos)){
        std::cout << "There are only ["
                  << qualifierCount << "] qualifiers (asking for ["
                  << pos << "])\n";
        throw 1;
      }

      return qualifiers[pos];
    }

    void _qualifierInfo::remove(const std::string &qName){
      for(int i = 0; i < qualifierCount; ++i){
        if(qualifiers[i] == qName){
          remove(i);
          return;
        }
      }
    }

    void _qualifierInfo::remove(const int pos,
                                const int count){
      for(int i = (pos + count); i < qualifierCount; ++i)
        qualifiers[i - count] = qualifiers[i];

      qualifierCount -= count;
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
    //============================================


    //---[ Type Info Class ]----------------------
    _typeInfo::_typeInfo() :
      leftQualifiers(),

      name(""),

      nestedInfoCount(0),
      nestedInfoIsType(NULL),
      nestedInfos(NULL),

      typedefing(NULL),
      baseType(NULL),

      typedefVar(NULL) {}

    _typeInfo::_typeInfo(const _typeInfo &type) :
      leftQualifiers(type.leftQualifiers),

      name(type.name),

      nestedInfoCount(type.nestedInfoCount),
      nestedInfoIsType(type.nestedInfoIsType),
      nestedInfos(type.nestedInfos),

      typedefing(type.typedefing),
      baseType(type.baseType),

      typedefVar(type.typedefVar)  {}

    _typeInfo& _typeInfo::operator = (const _typeInfo &type){
      leftQualifiers = type.leftQualifiers;

      name = type.name;

      nestedInfoCount  = type.nestedInfoCount;
      nestedInfoIsType = type.nestedInfoIsType;
      nestedInfos      = type.nestedInfos;

      typedefing = type.typedefing;
      baseType   = type.baseType;

      typedefVar = type.typedefVar;

      return *this;
    }

    strNode* _typeInfo::loadFrom(statement &s,
                                 strNode *nodePos){
      if(nodePos == NULL)
        return NULL;

      nodePos = leftQualifiers.loadFrom(s, nodePos);

      const bool hasTypedef = leftQualifiers.has("typedef");

      if(hasTypedef){
        _qualifierInfo newQuals = leftQualifiers.clone();
        newQuals.remove("typedef");

        leftQualifiers.remove(1, (leftQualifiers.qualifierCount - 1));

        typedefing = new _typeInfo;
        nodePos = typedefing->loadFrom(s, nodePos);

        typedefing->leftQualifiers = newQuals;
        baseType = typedefing;

        _varInfo typedefVarInfo;
        typedefVarInfo.baseType = NULL; //typedefing;

        typedefVar = new _varInfo;
        nodePos = typedefVar->loadFrom(s, nodePos, &typedefVarInfo);

        return nodePos;
      }

      if(nodePos &&
         (nodePos->type & unknownVariable)){

        if(s.hasTypeInScope(nodePos->value))
          name = nodePos->value;
        else // Type is temporary?
          name = nodePos->value;

        nodePos = nodePos->right;
      }

      if(nodePos &&
         (nodePos->type == startBrace)){
        strNode *nextNode = nodePos->right;
        nodePos = nodePos->down;

        const bool usesSemicolon = !leftQualifiers.has("enum");
        const char delimiter = (usesSemicolon ? ';' : ',');

        if(usesSemicolon)
          nestedInfoCount = statementCountWithDelimeter(nodePos, ';');
        else
          nestedInfoCount = statementCountWithDelimeter(nodePos, ',');

        nestedInfoIsType = new bool[nestedInfoCount];
        nestedInfos      = new typeOrExp[nestedInfoCount];

        for(int i = 0; i < nestedInfoCount; ++i){
          nestedInfoIsType[i] = (usesSemicolon                    ?
                                 statementIsATypeInfo(s, nodePos) :
                                 false);

          if(nestedInfoIsType[i]){
            nestedInfos[i].type = new _typeInfo;
            nodePos = nestedInfos[i].type->loadFrom(s, nodePos);
          }
          else{
            nestedInfos[i].exp = new expNode;

            statement *s2 = s.clone();
            s2->loadFromNode(nodePos);

            expNode::swap(*(nestedInfos[i].exp), s2->expRoot);

            while(nodePos &&
                  ((nodePos->value.size()) &&
                   (nodePos->value[0] != delimiter))){

              nodePos = nodePos->right;
            }

            if(nodePos)
              nodePos = nodePos->right;
          }
        }

        nodePos = nextNode;
      }

      return nodePos;
    }

    int _typeInfo::statementCountWithDelimeter(strNode *nodePos,
                                               const char delimiter){
      if(nodePos == NULL)
        return 0;

      int count = 0;

      while(nodePos){
        if(nodePos->value.size() &&
           (nodePos->value[0] == delimiter))
          ++count;

        nodePos = nodePos->right;
      }

      return count;
    }

    bool _typeInfo::statementIsATypeInfo(statement &s,
                                         strNode *nodePos){
      if(nodePos == NULL)
        return false;

      _qualifierInfo qualifiers;

      nodePos = qualifiers.loadFrom(s, nodePos);

      if(qualifiers.has("typedef"))
        return true;

      if(nodePos                           &&
         (nodePos->type & unknownVariable) &&
         (!s.hasTypeInScope(nodePos->value))){

        return true;
      }

      if(nodePos &&
         (nodePos->type == startBrace)){

        return true;
      }

      return false;
    }

    std::string _typeInfo::toString(const std::string &tab){
      std::string ret;

      if(typedefing &&
         (typedefing == baseType)){
        ret += tab;
        ret += "typedef ";
        ret += typedefing->toString();
        ret += ' ';
        ret += typedefVar->toString();
        ret += ';';
      }
      else{
        ret += tab;
        ret += leftQualifiers.toString();
        ret += name;

        if(nestedInfoCount){
          ret += '{';
          ret += '\n';

          for(int i = 0; i < nestedInfoCount; ++i){
            if(nestedInfoIsType[i]){
              ret += nestedInfos[i].type->toString(tab + "  ");
            }
            else {
              ret += nestedInfos[i].exp->getString(tab + "  ");
              ret += ';';
            }

            ret += '\n';
          }

          ret += tab;
          ret += '}';
        }
      }

      return ret;
    }

    _typeInfo::operator std::string (){
      return toString();
    }

    std::ostream& operator << (std::ostream &out, _typeInfo &type){
      out << type.toString();

      return out;
    }
    //============================================


    //---[ Variable Info Class ]------------------
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

      functionNestCount(0),
      functionNests(NULL) {}

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

      functionNestCount(var.functionNestCount),
      functionNests(var.functionNests) {}

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

      functionNestCount = var.functionNestCount;
      functionNests     = var.functionNests;

      return *this;
    }

    int _varInfo::variablesInStatement(strNode *nodePos){
      int argc = 0;

      while(nodePos){
        if((nodePos->value == ",") ||
           (nodePos->value == ";")){

          ++argc;
        }
        else if((nodePos->right) == NULL)
          ++argc;

        nodePos = nodePos->right;
      }

      return argc;
    }

    strNode* _varInfo::loadFrom(statement &s,
                                strNode *nodePos,
                                _varInfo *varHasType){
      nodePos = loadTypeFrom(s, nodePos, varHasType);

      info = getVarInfoFrom(s, nodePos);

      if(info & _varType::functionPointer){
        functionNestCount = getNestCountFrom(s, nodePos);
        functionNests     = new _varInfo[functionNestCount];
      }

      nodePos = loadNameFrom(s, nodePos);
      nodePos = loadArgsFrom(s, nodePos);

      if(nodePos &&
         (nodePos->value == ","))
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
      // No name var (argument for function)
      if(nodePos == NULL)
        return _varType::var;

      strNode *nextNode = nodePos->right;

      const int nestCount = getNestCountFrom(s, nodePos);

      if(nestCount)
        return _varType::functionPointer;

      if(nextNode &&
         (nextNode->type == startParentheses))
        return _varType::function;

      return _varType::var;
    }

    int _varInfo::getNestCountFrom(statement &s,
                                   strNode *nodePos){
      int nestCount = 0;

      while(nodePos &&
            (nodePos->type == startParentheses)){

        nodePos = nodePos->down;

        if(nodePos &&
           nodePos->value == "*"){

          ++nestCount;
          nodePos = nodePos->right;
        }
      }

      return nestCount;
    }

    strNode* _varInfo::loadNameFrom(statement &s,
                                    strNode *nodePos){
      if(nodePos == NULL)
        return NULL;

      strNode *nextNode = nodePos->right;

      int nestPos = 0;

      while(nodePos &&
            (nodePos->type == startParentheses)){

        nodePos = nodePos->down;

        if(nodePos &&
           nodePos->value == "*"){

          nodePos = nodePos->right;

          if(nodePos        &&
             nodePos->right &&
             (nodePos->right->type == startParentheses)){

            functionNests[nestPos].info = _varType::function;
            functionNests[nestPos].loadArgsFrom(s, nodePos->right);
          }

          ++nestPos;
        }
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

      if(nodePos->down){
        nodePos = nodePos->down;

        argumentCount    = variablesInStatement(nodePos);
        argumentVarInfos = new _varInfo[argumentCount];

        for(int i = 0; i < argumentCount; ++i)
          nodePos = argumentVarInfos[i].loadFrom(s, nodePos);
      }

      return nextNode;
    }

    //---[ Variable Info ]------------
    bool _varInfo::hasQualifier(const std::string &qName){
      return leftQualifiers.has(qName);
    }

    const std::string& _varInfo::getLeftQualifier(const int pos){
      return leftQualifiers.get(pos);
    }

    const std::string& _varInfo::getRightQualifier(const int pos){
      return rightQualifiers.get(pos);
    }

    const std::string& _varInfo::getLastLeftQualifier(){
      return leftQualifiers.get(leftQualifiers.qualifierCount - 1);
    }

    const std::string& _varInfo::getLastRightQualifier(){
      return rightQualifiers.get(rightQualifiers.qualifierCount - 1);
    }
    //================================

    std::string _varInfo::toString(const bool printType){
      std::string ret;

      bool addSpaceBeforeName = false;

      if(printType){
        ret += leftQualifiers.toString();

        if(baseType)
          ret += baseType->typeName;

        addSpaceBeforeName = !((rightQualifiers.qualifierCount) ||
                               (name.size()));

        if(!addSpaceBeforeName){
          if((info & _varType::functionType)  &&
             (rightQualifiers.qualifierCount) &&
             (getLastRightQualifier() == "*")){

            addSpaceBeforeName = true;
          }
        }

        if(!addSpaceBeforeName && baseType)
          ret += ' ';
      }

      ret += rightQualifiers.toString();

      for(int i = 0; i < functionNestCount; ++i)
        ret += "(*";

      if(addSpaceBeforeName &&
         (name.size() != 0))
        ret += ' ';

      ret += name;

      for(int i = 0; i < stackPointerCount; ++i){
        ret += '[';
        ret += (std::string) stackExpRoots[i];
        ret += ']';
      }

      for(int i = (functionNestCount - 1); 0 <= i; --i){
        ret += functionNests[i].toString();
        ret += ')';
      }

      if(info & _varType::functionType){
        ret += '(';

        for(int i = 0; i < argumentCount; ++i)
          ret += argumentVarInfos[i].toString();

        ret += ')';
      }

      return ret;
    }

    _varInfo::operator std::string (){
      return toString();
    }

    std::ostream& operator << (std::ostream &out, _varInfo &var){
      out << var.toString();
      return out;
    }
    //============================================

    void test(){
      parser p;
      p.loadLanguageTypes();
      statement &s = *(p.globalScope);

      strNode *nodeRoot = p.splitAndPreprocessContent("typedef struct a { int b, c; struct b {};} *b2;");
      // strNode *nodeRoot = p.splitAndPreprocessContent("const int * func(){}");

      // strNode *nodeRoot = p.splitAndPreprocessContent("const int *const ** const***a[2], *b, ((c)), d[3], e(int), (f), ((*g))(), (*(*h)(int))(double), (*(*(*i)())(int))(double);");

#if 0
      const int varCount = _varInfo::variablesInStatement(nodeRoot);

      if(varCount){
        _varInfo *variables = new _varInfo[varCount];

        nodeRoot = variables[0].loadFrom(s, nodeRoot);
        std::cout << "variables[0] = " << variables[0] << '\n';

        for(int i = 1; i < varCount; ++i){
          nodeRoot = variables[i].loadFrom(s, nodeRoot, &(variables[0]));
          std::cout << "variables[" << i << "] = " << variables[i] << '\n';
        }
      }
#else
      _typeInfo type;
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
