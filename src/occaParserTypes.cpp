#include "occaParserTypes.hpp"
#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    //---[ Qualifier Info Class ]-----------------
    qualifierInfo::qualifierInfo() :
      qualifierCount(0),
      qualifiers(NULL) {}

    qualifierInfo::qualifierInfo(const qualifierInfo &q) :
      qualifierCount(q.qualifierCount),
      qualifiers(q.qualifiers) {}

    qualifierInfo& qualifierInfo::operator = (const qualifierInfo &q){
      qualifierCount = q.qualifierCount;
      qualifiers     = q.qualifiers;

      return *this;
    }

    void qualifierInfo::free(){
      if(qualifiers)
        delete [] qualifiers;
    }

    qualifierInfo qualifierInfo::clone(){
      qualifierInfo q;

      q.qualifierCount = qualifierCount;

      q.qualifiers = new std::string[qualifierCount];

      for(int i = 0; i < qualifierCount; ++i)
        q.qualifiers[i] = qualifiers[i];

      return q;
    }

    int qualifierInfo::loadFrom(expNode &expRoot,
                                int leafPos){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      const int leafRoot = leafPos;

      while((leafPos < expRoot.leafCount) &&
            (expRoot.leaves[leafPos]->info & expType::qualifier)){

        ++leafPos;
      }

      qualifierCount = (leafPos - leafRoot);

      if(qualifierCount){
        qualifiers = new std::string[qualifierCount];

        for(int i = 0; i < qualifierCount; ++i)
          qualifiers[i] = expRoot.leaves[leafRoot + i]->value;
      }

      return leafPos;
    }

    strNode* qualifierInfo::loadFrom(statement &s,
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

    //---[ Qualifier Info ]-------------
    bool qualifierInfo::has(const std::string &qName) const {
      for(int i = 0; i < qualifierCount; ++i){
        if(qualifiers[i] == qName)
          return true;
      }

      return false;
    }

    const std::string& qualifierInfo::get(const int pos) const {
      if((pos < 0) ||
         (qualifierCount <= pos)){
        std::cout << "There are only ["
                  << qualifierCount << "] qualifiers (asking for ["
                  << pos << "])\n";
        throw 1;
      }

      return qualifiers[pos];
    }

    void qualifierInfo::add(const std::string &qName,
                            int pos){
      std::string *newQualifiers = new std::string[qualifierCount + 1];

      if(pos < 0)
        pos = qualifierCount;

      for(int i = 0; i < pos; ++i)
        newQualifiers[i] = qualifiers[i];

      newQualifiers[pos] = qName;

      for(int i = (pos + 1); i < qualifierCount; ++i)
        newQualifiers[i] = qualifiers[i - 1];

      delete [] qualifiers;

      qualifiers = newQualifiers;
    }

    void qualifierInfo::remove(const std::string &qName){
      for(int i = 0; i < qualifierCount; ++i){
        if(qualifiers[i] == qName){
          remove(i);
          return;
        }
      }
    }

    void qualifierInfo::remove(const int pos,
                               const int count){
      for(int i = (pos + count); i < qualifierCount; ++i)
        qualifiers[i - count] = qualifiers[i];

      qualifierCount -= count;
    }
    //==================================

    std::string qualifierInfo::toString(){
      std::string ret;

      for(int i = 0; i < qualifierCount; ++i){
        ret += qualifiers[i];

        if(((qualifiers[i][0] != '*') &&
            (qualifiers[i][0] != '&')) ||
           ( ((i + 1) < qualifierCount) &&
             ((qualifiers[i + 1][0] != '*') &&
              (qualifiers[i + 1][0] != '&')))){

          ret += ' ';
        }
      }

      return ret;
    }

    qualifierInfo::operator std::string (){
      return toString();
    }

    std::ostream& operator << (std::ostream &out, qualifierInfo &q){
      out << q.toString();

      return out;
    }
    //============================================


    //---[ Type Info Class ]----------------------
    typeInfo::typeInfo() :
      leftQualifiers(),

      name(""),

      nestedInfoCount(0),
      nestedInfoIsType(NULL),
      nestedInfos(NULL),

      typedefing(NULL),
      baseType(NULL),

      typedefVar(NULL) {}

    typeInfo::typeInfo(const typeInfo &type) :
      leftQualifiers(type.leftQualifiers),

      name(type.name),

      nestedInfoCount(type.nestedInfoCount),
      nestedInfoIsType(type.nestedInfoIsType),
      nestedInfos(type.nestedInfos),

      typedefing(type.typedefing),
      baseType(type.baseType),

      typedefVar(type.typedefVar)  {}

    typeInfo& typeInfo::operator = (const typeInfo &type){
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

    //---[ NEW ]--------------
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

        nestedInfoCount = delimeterCount(leaf, delimiter);
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

            nestedExps[i].organize();

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

    int typeInfo::delimeterCount(expNode &expRoot,
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
    //========================

    strNode* typeInfo::loadFrom(statement &s,
                                strNode *nodePos){
      if(nodePos == NULL)
        return NULL;

      nodePos = leftQualifiers.loadFrom(s, nodePos);

      const bool hasTypedef = leftQualifiers.has("typedef");

      if(hasTypedef){
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
        const char *delimiter    = (usesSemicolon ? ";" : ",");

        nestedInfoCount  = statementCountWithDelimeter(nodePos, delimiter);
        nestedInfoIsType = new bool[nestedInfoCount];
        nestedInfos      = new typeOrVar[nestedInfoCount];

        for(int i = 0; i < nestedInfoCount; ++i){
          nestedInfoIsType[i] = (usesSemicolon                    ?
                                 statementIsATypeInfo(s, nodePos) :
                                 false);

          if(nestedInfoIsType[i]){
            nestedInfos[i].type = new typeInfo;
            nodePos = nestedInfos[i].type->loadFrom(s, nodePos);
          }
          else{
            nestedInfos[i].varLeaf      = new varLeaf_t;
            nestedInfos[i].varLeaf->var = new varInfo;
            nodePos = nestedInfos[i].varLeaf->var->loadFrom(s, nodePos);

            if(nodePos->value == "="){
              nestedInfos[i].varLeaf->exp = new expNode;

              s.setExpNodeFromStrNode(*(nestedInfos[i].varLeaf->exp),
                                      nodePos->left);

              while(nodePos &&
                    (nodePos->value != delimiter)){

                nodePos = nodePos->right;
              }
            }

            if(nodePos)
              nodePos = nodePos->right;
          }
        }

        nodePos = nextNode;
      }

      return nodePos;
    }

    int typeInfo::statementCountWithDelimeter(strNode *nodePos,
                                              const char *delimiter){
      if(nodePos == NULL)
        return 0;

      int count = 0;

      while(nodePos){
        if((nodePos->value == delimiter))
          ++count;

        nodePos = nodePos->right;
      }

      return count;
    }

    bool typeInfo::statementIsATypeInfo(statement &s,
                                        strNode *nodePos){
      if(nodePos == NULL)
        return false;

      qualifierInfo qualifiers;

      nodePos = qualifiers.loadFrom(s, nodePos);

      if(qualifiers.has("typedef")){
        qualifiers.free();
        return true;
      }

      if(nodePos                           &&
         (nodePos->type & unknownVariable) &&
         (!s.hasTypeInScope(nodePos->value))){

        qualifiers.free();
        return true;
      }

      if(nodePos &&
         (nodePos->type == startBrace)){

        qualifiers.free();
        return true;
      }

      qualifiers.free();
      return false;
    }

    bool typeInfo::statementIsATypeInfo(expNode &expRoot,
                                        int leafPos){
      if(expRoot.leafCount == 0)
        return false;

      qualifierInfo qualifiers;

      leafPos = qualifiers.loadFrom(expRoot, leafPos);

      if(qualifiers.has("typedef"))
        return true;

      if(leafPos < expRoot.leafCount){
        if((expRoot[leafPos].info & expType::unknown) &&
           (!expRoot.sInfo->hasTypeInScope(expRoot[leafPos].value))){

          return true;
        }

        if(expRoot[leafPos].value == "{")
          return true;
      }

      return false;
    }

    //---[ Type Info ]------------------
    void typeInfo::addQualifier(const std::string &qName,
                                int pos){
      leftQualifiers.add(qName, pos);
    }
    //==================================

    std::string typeInfo::toString(const std::string &tab){
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
          if(name.size())
            ret += ' ';

          ret += '{';
          ret += '\n';

          for(int i = 0; i < nestedInfoCount; ++i){
#if 1
            ret += nestedExps[i].getString(tab + "  ");
#else
            if(nestedInfoIsType[i]){
              ret += nestedInfos[i].type->toString(tab + "  ");
            }
            else {
              ret += tab + "  ";
              ret += nestedInfos[i].varLeaf->var->toString();

              if(nestedInfos[i].varLeaf->hasExp)
                ret += nestedInfos[i].varLeaf->exp->getString();

              ret += ';';
            }
#endif

            ret += '\n';
          }

          ret += tab;
          ret += '}';
        }
      }

      return ret;
    }

    typeInfo::operator std::string (){
      return toString();
    }

    std::ostream& operator << (std::ostream &out, typeInfo &type){
      out << type.toString();

      return out;
    }
    //============================================


    //---[ Variable Info Class ]------------------
    varInfo::varInfo() :
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

    varInfo::varInfo(const varInfo &var) :
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

    varInfo& varInfo::operator = (const varInfo &var){
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

    int varInfo::variablesInStatement(strNode *nodePos){
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

    //---[ NEW ]--------------
    int varInfo::loadFrom(expNode &expRoot,
                          int leafPos,
                          varInfo *varHasType){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      leafPos = loadTypeFrom(expRoot, leafPos, varHasType);

      info = getVarInfoFrom(expRoot, leafPos);

      if(info & varType::functionPointer){
        functionNestCount = getNestCountFrom(expRoot, leafPos);
        functionNests     = new varInfo[functionNestCount];
      }

      leafPos = loadNameFrom(expRoot, leafPos);
      leafPos = loadArgsFrom(expRoot, leafPos);

      return leafPos;
    }

    int varInfo::loadTypeFrom(expNode &expRoot,
                              int leafPos,
                              varInfo *varHasType){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      if(varHasType == NULL){
        leafPos = leftQualifiers.loadFrom(expRoot, leafPos);

        if(leafPos < expRoot.leafCount){
          baseType = expRoot.sInfo->hasTypeInScope(expRoot[leafPos].value);

          if(baseType)
            ++leafPos;
        }
      }
      else{
        leftQualifiers = varHasType->leftQualifiers;
        baseType       = varHasType->baseType;
      }

      leafPos = rightQualifiers.loadFrom(expRoot, leafPos);

      return leafPos;
    }

    int varInfo::getVarInfoFrom(expNode &expRoot,
                                int leafPos){
      // No name var (argument for function)
      if(expRoot.leafCount <= leafPos)
        return varType::var;

      const int nestCount = getNestCountFrom(expRoot, leafPos);

      if(nestCount)
        return varType::functionPointer;

      ++leafPos;

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "(")){

        ++leafPos;

        if((leafPos < expRoot.leafCount) &&
           (expRoot[leafPos].value == "{")){

          return varType::functionDef;
        }
        else{
          return varType::functionDec;
        }
      }

      return varType::var;
    }

    int varInfo::getNestCountFrom(expNode &expRoot,
                                  int leafPos){
      if(expRoot.leafCount <= leafPos)
        return 0;

      int nestCount = 0;

      expNode *leaf = expRoot.leaves[leafPos];

      while((leaf->value == "(") &&
            (leaf->leafCount != 0)){

        if(leaf->leaves[0]->value == "*"){
          ++nestCount;

          if(1 < leaf->leafCount)
            leaf = leaf->leaves[1];
          else
            break;
        }
        else
          leaf = leaf->leaves[0];
      }

      return nestCount;
    }

    int varInfo::loadNameFrom(expNode &expRoot,
                              int leafPos){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      int nestPos = 0;

      expNode *leaf = expRoot.leaves[leafPos];

      if(leaf->value == "(")
        ++leafPos;

      while((leaf->value == "(") &&
            (leaf->leafCount != 0)){

        if(leaf->leaves[0]->value == "*"){
          ++nestPos;

          if(1 < leaf->leafCount){
            if((2 < leaf->leafCount) &&
               (leaf->leaves[2]->value == "(")){

              functionNests[nestPos - 1].info = varType::function;
              functionNests[nestPos - 1].loadArgsFrom(*leaf, 2);
            }

            leaf = leaf->leaves[1];
          }
          else
            break;
        }
        else
          leaf = leaf->leaves[0];
      }

      if(leaf->info & expType::unknown){
        name = leaf->value;

        int sLeafPos = leaf->whichLeafAmI();

        if(leaf->up == &expRoot){
          return loadStackPointersFrom(expRoot, sLeafPos + 1);
        }
        else
          loadStackPointersFrom(*leaf, sLeafPos + 1);
      }

      return leafPos;
    }

    int varInfo::loadStackPointersFrom(expNode &expRoot,
                                       int leafPos){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      for(int i = leafPos; i < expRoot.leafCount; ++i){
        if(expRoot[i].value == "[")
          ++stackPointerCount;
        else
          break;
      }

      if(stackPointerCount){
        stackExpRoots = new expNode[stackPointerCount];

        for(int i = 0; i < stackPointerCount; ++i)
          expNode::swap(stackExpRoots[i], expRoot[leafPos + i]);
      }

      return (leafPos + stackPointerCount);
    }

    int varInfo::loadArgsFrom(expNode &expRoot,
                              int leafPos){
      if( !(info & varType::functionType) )
        return leafPos;

      if(expRoot.leafCount <= leafPos){
        std::cout << "Missing arguments from function variable\n";
        throw 1;
      }

      if(expRoot[leafPos].leafCount){
        expNode &leaf = expRoot[leafPos];
        int sLeafPos  = 0;

        argumentCount    = 1 + typeInfo::delimeterCount(leaf, ",");
        argumentVarInfos = new varInfo[argumentCount];

        for(int i = 0; i < argumentCount; ++i){
          sLeafPos = argumentVarInfos[i].loadFrom(leaf, sLeafPos);
          sLeafPos = typeInfo::nextDelimeter(leaf, sLeafPos, ",") + 1;
        }

        return (leafPos + 1);
      }

      return leafPos;
    }
    //========================

    strNode* varInfo::loadFrom(statement &s,
                               strNode *nodePos,
                               varInfo *varHasType){
      nodePos = loadTypeFrom(s, nodePos, varHasType);

      info = getVarInfoFrom(s, nodePos);

      if(info & varType::functionPointer){
        functionNestCount = getNestCountFrom(s, nodePos);
        functionNests     = new varInfo[functionNestCount];
      }

      nodePos = loadNameFrom(s, nodePos);
      nodePos = loadArgsFrom(s, nodePos);

      if(nodePos &&
         (nodePos->value == ","))
        nodePos = nodePos->right;

      return nodePos;
    }

    strNode* varInfo::loadTypeFrom(statement &s,
                                   strNode *nodePos,
                                   varInfo *varHasType){
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

    int varInfo::getVarInfoFrom(statement &s,
                                strNode *nodePos){
      // No name var (argument for function)
      if(nodePos == NULL)
        return varType::var;

      strNode *nextNode = nodePos->right;

      const int nestCount = getNestCountFrom(s, nodePos);

      if(nestCount)
        return varType::functionPointer;

      if(nextNode &&
         (nextNode->type == startParentheses)){

        if((nextNode->right)       &&
           (nextNode->right->type == startBrace)){

          return varType::functionDef;
        }
        else{
          return varType::functionDec;
        }
      }

      return varType::var;
    }

    int varInfo::getNestCountFrom(statement &s,
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

    strNode* varInfo::loadNameFrom(statement &s,
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

            functionNests[nestPos].info = varType::function;
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

    strNode* varInfo::loadStackPointersFrom(statement &s,
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

          if(nodePos->down)
            s.setExpNodeFromStrNode(stackExpRoots[i], nodePos->down);

          nodePos = nodePos->right;
        }
      }

      return nodePos;
    }

    strNode* varInfo::loadArgsFrom(statement &s,
                                   strNode *nodePos){
      if( !(info & varType::functionType) )
        return nodePos;

      if(nodePos == NULL){
        std::cout << "Missing arguments from function variable\n";
        throw 1;
      }

      strNode *nextNode = nodePos->right;

      if(nodePos->down){
        nodePos = nodePos->down;

        argumentCount    = variablesInStatement(nodePos);
        argumentVarInfos = new varInfo[argumentCount];

        for(int i = 0; i < argumentCount; ++i)
          nodePos = argumentVarInfos[i].loadFrom(s, nodePos);
      }

      return nextNode;
    }

    //---[ Variable Info ]------------
    int varInfo::leftQualifierCount() const {
      return leftQualifiers.qualifierCount;
    }

    int varInfo::rightQualifierCount() const {
      return rightQualifiers.qualifierCount;
    }

    bool varInfo::hasQualifier(const std::string &qName) const {
      return leftQualifiers.has(qName);
    }

    void varInfo::addQualifier(const std::string &qName,
                               int pos){
      leftQualifiers.add(qName, pos);
    }

    void varInfo::removeQualifier(const std::string &qName){
      leftQualifiers.remove(qName);
    }

    const std::string& varInfo::getLeftQualifier(const int pos) const {
      return leftQualifiers.get(pos);
    }

    const std::string& varInfo::getRightQualifier(const int pos) const {
      return rightQualifiers.get(pos);
    }

    const std::string& varInfo::getLastLeftQualifier() const {
      return leftQualifiers.get(leftQualifiers.qualifierCount - 1);
    }

    const std::string& varInfo::getLastRightQualifier() const {
      return rightQualifiers.get(rightQualifiers.qualifierCount - 1);
    }
    //================================

    std::string varInfo::toString(const bool printType){
      std::string ret;

      bool addSpaceBeforeName = false;

      if(printType){
        ret += leftQualifiers.toString();

        if(baseType)
          ret += baseType->name;

        addSpaceBeforeName = !((rightQualifiers.qualifierCount) ||
                               (name.size()));

        if(!addSpaceBeforeName){
          if((info & varType::functionType)  &&
             (rightQualifiers.qualifierCount) &&
             ((getLastRightQualifier() == "*") ||
              (getLastRightQualifier() == "&"))){

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

      for(int i = 0; i < stackPointerCount; ++i)
        ret += (std::string) stackExpRoots[i];

      for(int i = (functionNestCount - 1); 0 <= i; --i){
        ret += functionNests[i].toString();
        ret += ')';
      }

      if(info & varType::functionType){
        ret += '(';

        if(argumentCount){
          ret += argumentVarInfos[0].toString();

          for(int i = 1; i < argumentCount; ++i){
            ret += ", ";
            ret += argumentVarInfos[i].toString();
          }
        }

        ret += ')';
      }

      return ret;
    }

    varInfo::operator std::string (){
      return toString();
    }

    std::ostream& operator << (std::ostream &out, varInfo &var){
      out << var.toString();
      return out;
    }
    //============================================
  };
};
