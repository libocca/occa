#include "occaParserTypes.hpp"
#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
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
      if(qualifiers){
        qualifierCount = 0;

        delete [] qualifiers;
        qualifiers = NULL;
      }
    }

    qualifierInfo qualifierInfo::clone(){
      qualifierInfo q;

      q.qualifierCount = qualifierCount;

      if(qualifierCount){
        q.qualifiers = new std::string[qualifierCount];

        for(int i = 0; i < qualifierCount; ++i)
          q.qualifiers[i] = qualifiers[i];
      }

      return q;
    }

    int qualifierInfo::loadFrom(expNode &expRoot,
                                int leafPos){

      if(expRoot.sInfo == NULL)
        throw 1;

      return loadFrom(*(expRoot.sInfo), expRoot, leafPos);
    }

    int qualifierInfo::loadFrom(statement &s,
                                expNode &expRoot,
                                int leafPos){

      if(expRoot.leafCount <= leafPos)
        return leafPos;

      const int leafRoot = leafPos;

      while((leafPos < expRoot.leafCount) &&
            (expRoot[leafPos].info & expType::qualifier)){

        ++leafPos;
      }

      qualifierCount = (leafPos - leafRoot);

      if(qualifierCount){
        qualifiers = new std::string[qualifierCount];

        for(int i = 0; i < qualifierCount; ++i)
          qualifiers[i] = expRoot[leafRoot + i].value;
      }

      return leafPos;
    }

    int qualifierInfo::loadFromFortran(statement &s,
                                       varInfo &var,
                                       expNode &expRoot,
                                       int leafPos){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      while(true){
        int newLeafPos = updateFortranVar(s, var, expRoot, leafPos);

        if(newLeafPos == leafPos)
          break;

        if(leafPos < expRoot.leafCount){
          if(expRoot[newLeafPos].value == ","){
            ++newLeafPos;
          }
          else if(expRoot[newLeafPos].value == "::"){
            leafPos = newLeafPos + 1;
            break;
          }
        }

        leafPos = newLeafPos;
      }

      return leafPos;
    }

    bool qualifierInfo::fortranVarNeedsUpdate(varInfo &var,
                                              const std::string &fortranQualifier){
      // Normal Fortran
      if(fortranQualifier == "POINTER"){
        ++(var.pointerCount);
        return true;
      }
      else if(fortranQualifier == "VOLATILE"){
        add("volatile");
        return true;
      }
      else if(fortranQualifier == "PARAMETER"){
        add("const", 0);
        return true;
      }
      // OFL Keywords
      if(fortranQualifier == "KERNEL"){
        add("occaKernel");
        return true;
      }
      if(fortranQualifier == "DEVICE"){
        add("occaFunction");
        return true;
      }
      if(fortranQualifier == "SHARED"){
        add("occaShared");
        return true;
      }
      if(fortranQualifier == "EXCLUSIVE"){
        add("exclusive");
        return true;
      }

      return false;
    }

    int qualifierInfo::updateFortranVar(statement &s,
                                        varInfo &var,
                                        expNode &expPos,
                                        const int leafPos){
      if(fortranVarNeedsUpdate(var, expPos[leafPos]))
        return (leafPos + 1);

      if(expPos[leafPos].info & expType::type){
        int nextLeafPos = leafPos;

        std::string typeName = varInfo::getFullFortranType(expPos, nextLeafPos);
        var.baseType = s.hasTypeInScope(typeName);

        return nextLeafPos;
      }
      else{
        const std::string &value = expPos[leafPos].value;

        if(value == "INTENT"){
          expNode *leaf = expPos.leaves[leafPos + 1];

          if(leaf && (leaf->leafCount)){
            leaf = leaf->leaves[0];

            var.leftQualifiers.add("INTENT" + upString(leaf->value));
            var.rightQualifiers.add("&", 0);

            if(upStringCheck(leaf->value, "IN"))
              add("const", 0);

            return (leafPos + 2);
          }

          return (leafPos + 1);
        }
        else if(value == "DIMENSION"){
          var.leftQualifiers.add("DIMENSION");
          return var.loadStackPointersFromFortran(expPos, leafPos + 1);
        }
      }

      return leafPos;
    }

    //---[ Qualifier Info ]-------------
    bool qualifierInfo::has(const std::string &qName){
      for(int i = 0; i < qualifierCount; ++i){
        if(qualifiers[i] == qName)
          return true;
      }

      return false;
    }

    std::string& qualifierInfo::get(const int pos){
      OCCA_CHECK((0 <= pos) && (pos < qualifierCount),
                 "There are only ["
                 << qualifierCount << "] qualifiers (asking for ["
                 << pos << "])");

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

      for(int i = pos; i < qualifierCount; ++i)
        newQualifiers[i + 1] = qualifiers[i];

      delete [] qualifiers;

      qualifiers = newQualifiers;
      ++qualifierCount;
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

      if((qualifierCount == 0) &&
         (count != 0)){

        delete [] qualifiers;
        qualifiers = NULL;
      }
    }

    void qualifierInfo::clear(){
      if(qualifierCount){
        qualifierCount = 0;

        delete [] qualifiers;
        qualifiers = NULL;
      }
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

    qualifierInfo::operator std::string () {
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
      nestedExps(NULL),

      typedefHasDefinition(false),
      typedefing(NULL),
      baseType(NULL),

      typedefVar(NULL) {}

    typeInfo::typeInfo(const typeInfo &type) :
      leftQualifiers(type.leftQualifiers),

      name(type.name),

      nestedInfoCount(type.nestedInfoCount),
      nestedExps(type.nestedExps),

      typedefHasDefinition(type.typedefHasDefinition),
      typedefing(type.typedefing),
      baseType(type.baseType),

      typedefVar(type.typedefVar)  {}

    typeInfo& typeInfo::operator = (const typeInfo &type){
      leftQualifiers = type.leftQualifiers;

      name = type.name;

      nestedInfoCount = type.nestedInfoCount;
      nestedExps      = type.nestedExps;

      typedefHasDefinition = type.typedefHasDefinition;
      typedefing           = type.typedefing;
      baseType             = type.baseType;

      typedefVar = type.typedefVar;

      return *this;
    }

    typeInfo typeInfo::clone(){
      typeInfo c = *this;

      c.leftQualifiers = leftQualifiers.clone();

      if(nestedInfoCount){
        c.nestedExps = new expNode[nestedInfoCount];

        for(int i = 0; i < nestedInfoCount; ++i)
          nestedExps[i].cloneTo(c.nestedExps[i]);
      }

      if(typedefVar){
        c.typedefVar  = new varInfo;
        *c.typedefVar = typedefVar->clone();
      }

      return c;
    }

    //---[ Load Info ]------------------
    int typeInfo::loadFrom(expNode &expRoot,
                           int leafPos){

      if(expRoot.sInfo == NULL)
        throw 1;

      return loadFrom(*(expRoot.sInfo), expRoot, leafPos);
    }

    int typeInfo::loadFrom(statement &s,
                           expNode &expRoot,
                           int leafPos){

      if(expRoot.leafCount <= leafPos)
        return leafPos;

      leafPos = leftQualifiers.loadFrom(s, expRoot, leafPos);

      if(leftQualifiers.has("typedef"))
        return loadTypedefFrom(s, expRoot, leafPos);

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].info & expType::unknown)){

        name = expRoot[leafPos++].value;
      }

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "{")){

        expNode &leaf = expRoot[leafPos++];

        const bool usesSemicolon = !leftQualifiers.has("enum");
        const char *delimiter = (usesSemicolon ? ";" : ",");

        nestedInfoCount = delimiterCount(leaf, delimiter);
        nestedExps      = new expNode[nestedInfoCount];

        int sLeafPos = 0;

        for(int i = 0; i < nestedInfoCount; ++i){
          int sNextLeafPos = nextDelimiter(leaf, sLeafPos, delimiter);

          // Empty statements
          if(sNextLeafPos != sLeafPos){
            const bool loadType = typeInfo::statementIsATypeInfo(s, leaf, sLeafPos);

            sNextLeafPos = leaf.mergeRange(expType::root,
                                           sLeafPos,
                                           sNextLeafPos);

            expNode::swap(nestedExps[i], leaf[sLeafPos]);

            if(!loadType)
              nestedExps[i].splitDeclareStatement(parsingFortran);
            else
              nestedExps[i].splitStructStatement(parsingFortran);

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

    int typeInfo::loadTypedefFrom(statement &s,
                                  expNode &expRoot,
                                  int leafPos){
      leftQualifiers.remove("typedef");

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value != "{")){
        typeInfo *tmp = s.hasTypeInScope(expRoot[leafPos].value);

        if(tmp){
          typedefing = tmp;
        }
        else{
          typedefing           = new typeInfo;
          typedefing->name     = expRoot[leafPos].value;
          typedefing->baseType = typedefing;
        }

        ++leafPos;
      }

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "{")){
        // Anonymous type
        if(typedefing == NULL){
          typedefing           = new typeInfo;
          typedefing->baseType = typedefing;
        }

        typedefing->loadFrom(s, expRoot, leafPos);
        ++leafPos;

        typedefHasDefinition = true;
      }

      baseType = typedefing->baseType;

      varInfo typedefVarInfo;
      typedefVarInfo.baseType = typedefing;

      typedefVar = new varInfo;
      leafPos = typedefVar->loadFrom(s, expRoot, leafPos, &typedefVarInfo);

      name = typedefVar->name;

      return leafPos;
    }

    bool typeInfo::statementIsATypeInfo(expNode &expRoot,
                                        int leafPos){

      if(expRoot.sInfo == NULL)
        throw 1;

      return statementIsATypeInfo(*(expRoot.sInfo), expRoot, leafPos);
    }

    bool typeInfo::statementIsATypeInfo(statement &s,
                                        expNode &expRoot,
                                        int leafPos){
      if(expRoot.leafCount == 0)
        return false;

      qualifierInfo qualifiers;

      leafPos = qualifiers.loadFrom(s, expRoot, leafPos);

      if(qualifiers.has("typedef"))
        return true;

      if(leafPos < expRoot.leafCount){
        if((expRoot[leafPos].info & expType::unknown) &&
           (!s.hasTypeInScope(expRoot[leafPos].value))){

          return true;
        }

        if(expRoot[leafPos].value == "{")
          return true;
      }

      return false;
    }

    int typeInfo::delimiterCount(expNode &expRoot,
                                 const char *delimiter){
      int count = 0;

      for(int i = 0; i < expRoot.leafCount; ++i){
        if(expRoot[i].value == delimiter)
          ++count;
      }

      return count;
    }

    int typeInfo::nextDelimiter(expNode &expRoot,
                                int leafPos,
                                const char *delimiter){
      for(int i = leafPos; i < expRoot.leafCount; ++i){
        if(expRoot[i].value == delimiter)
          return i;
      }

      return expRoot.leafCount;
    }
    //==================================


    //---[ Type Info ]------------------
    void typeInfo::addQualifier(const std::string &qName,
                                int pos){
      leftQualifiers.add(qName, pos);
    }
    //==================================

    std::string typeInfo::toString(const std::string &tab){
      std::string ret;

      if(typedefing){
        ret += tab;
        ret += "typedef ";
        ret += leftQualifiers.toString();

        if(typedefHasDefinition)
          ret += typedefing->toString();
        else
          ret += typedefing->name;

        ret += ' ';
        ret += typedefVar->toString(false);
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

          for(int i = 0; i < nestedInfoCount; ++i)
            ret += nestedExps[i].toString(tab + "  ");

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
      stackPointersUsed(0),
      stackExpRoots(NULL),

      usesTemplate(false),
      tArgCount(0),
      tArgs(NULL),

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
      stackPointersUsed(var.stackPointersUsed),
      stackExpRoots(var.stackExpRoots),

      usesTemplate(var.usesTemplate),
      tArgCount(var.tArgCount),
      tArgs(var.tArgs),

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

      pointerCount = var.pointerCount;

      stackPointerCount  = var.stackPointerCount;
      stackPointersUsed  = var.stackPointersUsed;
      stackExpRoots      = var.stackExpRoots;

      usesTemplate = var.usesTemplate;
      tArgCount    = var.tArgCount;
      tArgs        = var.tArgs;

      argumentCount    = var.argumentCount;
      argumentVarInfos = var.argumentVarInfos;

      functionNestCount = var.functionNestCount;
      functionNests     = var.functionNests;

      return *this;
    }

    varInfo varInfo::clone(){
      varInfo v = *this;

      v.leftQualifiers  = leftQualifiers.clone();
      v.rightQualifiers = rightQualifiers.clone();

      if(stackPointerCount){
        v.stackExpRoots = new expNode[stackPointerCount];

        for(int i = 0; i < stackPointerCount; ++i)
          stackExpRoots[i].cloneTo(v.stackExpRoots[i]);
      }

      if(tArgCount){
        v.tArgs = new typeInfo*[tArgCount];

        for(int i = 0; i < tArgCount; ++i)
          v.tArgs[i] = new typeInfo(tArgs[i]->clone());
      }

      if(argumentCount){
        v.argumentVarInfos = new varInfo*[argumentCount];

        for(int i = 0; i < argumentCount; ++i)
          v.argumentVarInfos[i] = new varInfo(argumentVarInfos[i]->clone());
      }

      if(functionNestCount){
        v.functionNests = new varInfo[functionNestCount];

        for(int i = 0; i < functionNestCount; ++i)
          v.functionNests[i] = functionNests[i].clone();
      }

      return v;
    }

    int varInfo::variablesInStatement(expNode &expRoot){
      int argc = 0;

      for(int i = 0; i < expRoot.leafCount; ++i){
        if((expRoot[i].value == ",") ||
           (expRoot[i].value == ";")){

          ++argc;
        }
        else if(i == (expRoot.leafCount - 1))
          ++argc;
      }

      return argc;
    }

    //---[ Load Info ]------------------
    int varInfo::loadFrom(expNode &expRoot,
                          int leafPos,
                          varInfo *varHasType){

      if(expRoot.sInfo == NULL)
        throw 1;

      return loadFrom(*(expRoot.sInfo), expRoot, leafPos, varHasType);
    }

    int varInfo::loadFrom(statement &s,
                          expNode &expRoot,
                          int leafPos,
                          varInfo *varHasType){

      if(expRoot.leafCount <= leafPos)
        return leafPos;

      leafPos = loadTypeFrom(s, expRoot, leafPos, varHasType);

      info = getVarInfoFrom(s, expRoot, leafPos);

      if(info & varType::functionPointer){
        functionNestCount = getNestCountFrom(expRoot, leafPos);
        functionNests     = new varInfo[functionNestCount];
      }

      leafPos = loadNameFrom(s, expRoot, leafPos);
      leafPos = loadArgsFrom(s, expRoot, leafPos);

      return leafPos;
    }

    int varInfo::loadTypeFrom(statement &s,
                              expNode &expRoot,
                              int leafPos,
                              varInfo *varHasType){

      if(expRoot.leafCount <= leafPos)
        return leafPos;

      if(varHasType == NULL){
        leafPos = leftQualifiers.loadFrom(s, expRoot, leafPos);

        if(leafPos < expRoot.leafCount){
          baseType = s.hasTypeInScope(expRoot[leafPos].value);

          if(baseType)
            ++leafPos;
        }
      }
      else{
        leftQualifiers = varHasType->leftQualifiers.clone();
        baseType       = varHasType->baseType;
      }

      leafPos = rightQualifiers.loadFrom(s, expRoot, leafPos);

      for(int i = 0; i < rightQualifiers.qualifierCount; ++i){
        if(rightQualifiers[i] == "*")
          ++pointerCount;
      }

      return leafPos;
    }

    int varInfo::getVarInfoFrom(statement &s,
                                expNode &expRoot,
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

    int varInfo::loadNameFrom(statement &s,
                              expNode &expRoot,
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
              functionNests[nestPos - 1].loadArgsFrom(s, *leaf, 2);
            }

            leaf = leaf->leaves[1];
          }
          else
            break;
        }
        else
          leaf = leaf->leaves[0];
      }

      if(leaf->info & (expType::unknown  |
                       expType::varInfo  |
                       expType::funcInfo)){

        if(leaf->info & expType::varInfo){
          if(baseType)
            name = leaf->getVarInfo().name;
          else
            return leafPos;
        }
        else
          name = leaf->value;

        int sLeafPos = leaf->whichLeafAmI();

        if(leaf->up == &expRoot){
          return loadStackPointersFrom(expRoot, sLeafPos + 1);
        }
        else{
          loadStackPointersFrom(*leaf, sLeafPos + 1);
        }
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

      stackPointersUsed = stackPointerCount;

      return (leafPos + stackPointerCount);
    }

    int varInfo::loadArgsFrom(statement &s,
                              expNode &expRoot,
                              int leafPos){
      if( !(info & varType::functionType) )
        return leafPos;

      OCCA_CHECK(leafPos < expRoot.leafCount,
                 "Missing arguments from function variable");

      if(expRoot[leafPos].leafCount){
        expNode &leaf = expRoot[leafPos];
        int sLeafPos  = 0;

        argumentCount    = 1 + typeInfo::delimiterCount(leaf, ",");
        argumentVarInfos = new varInfo*[argumentCount];

        for(int i = 0; i < argumentCount; ++i){
          argumentVarInfos[i] = new varInfo();
          sLeafPos = argumentVarInfos[i]->loadFrom(s, leaf, sLeafPos);
          sLeafPos = typeInfo::nextDelimiter(leaf, sLeafPos, ",") + 1;
        }
      }

      return (leafPos + 1);
    }

    //   ---[ Fortran ]-------
    int varInfo::loadFromFortran(expNode &expRoot,
                                 int leafPos,
                                 varInfo *varHasType){

      if(expRoot.sInfo == NULL)
        throw 1;

      return loadFromFortran(*(expRoot.sInfo), expRoot, leafPos, varHasType);
    }

    int varInfo::loadFromFortran(statement &s,
                                 expNode &expRoot,
                                 int leafPos,
                                 varInfo *varHasType){
      // Load Type
      leafPos = loadTypeFromFortran(s, expRoot, leafPos, varHasType);

      // Load Name
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      name = expRoot[leafPos++].value;

      // Load Args
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      if(expRoot[leafPos].leafCount){
        expNode &leaf = *(expRoot.leaves[leafPos]);

        if(info & varType::functionType){
          argumentCount = (leaf.leafCount + 1)/2;

          if(argumentCount)
            argumentVarInfos = new varInfo*[argumentCount];

          for(int i = 0; i < argumentCount; ++i){
            argumentVarInfos[i] = new varInfo();
            argumentVarInfos[i]->name = leaf[2*i].value;
          }

          leafPos = expRoot.leafCount;
        }
        else{
          leafPos = loadStackPointersFromFortran(expRoot, leafPos);
        }
      }

      return leafPos;
    }

    int varInfo::loadTypeFromFortran(expNode &expRoot,
                                     int leafPos,
                                     varInfo *varHasType){

      if(expRoot.sInfo == NULL)
        throw 1;

      return loadTypeFromFortran(*(expRoot.sInfo), expRoot, leafPos, varHasType);
    }

    int varInfo::loadTypeFromFortran(statement &s,
                                     expNode &expRoot,
                                     int leafPos,
                                     varInfo *varHasType){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      if(varHasType == NULL){
        leafPos = leftQualifiers.loadFromFortran(s, *this, expRoot, leafPos);

        if(leafPos < expRoot.leafCount){
          if(expRoot[leafPos].value == "SUBROUTINE"){
            baseType = s.hasTypeInScope("void");
            info    |= varType::functionDec;
            ++leafPos;
          }
          else if(expRoot[leafPos].value == "FUNCTION"){
            info |= varType::functionDec;
            ++leafPos;
          }
        }
      }
      else{
        leftQualifiers  = varHasType->leftQualifiers.clone();
        rightQualifiers = varHasType->rightQualifiers.clone();
        baseType        = varHasType->baseType;
      }

      if( !(info & varType::functionDec) )
        info |= varType::var;

      return leafPos;
    }

    std::string varInfo::getFullFortranType(expNode &expRoot,
                                            int &leafPos){
      if( !(expRoot[leafPos].info & expType::type) )
        return "";

      std::string typeNode = expRoot[leafPos++].value;

      if(leafPos < expRoot.leafCount){
        int bytes = -1;

        // [-] Ignoring complex case
        const bool isFloat = ((typeNode.find("REAL") != std::string::npos) ||
                              (typeNode == "PRECISION")                    ||
                              (typeNode == "COMPLEX"));

        const int typeNodeChars = typeNode.size();
        const bool typeHasSuffix = isANumber(typeNode[typeNodeChars - 1]);

        std::string suffix = "";

        if(typeHasSuffix){
          for(int i = 0; i < typeNodeChars; ++i){
            if(isANumber(typeNode[i]))
              suffix += typeNode[i];
          }
        }

        if(isFloat){
          if(typeNode.find("REAL") != std::string::npos)
            bytes = 4;
          else if(typeNode == "PRECISION")
            bytes = 8;
        }
        else {
          if(typeNode.find("INTEGER") != std::string::npos)
            bytes = 4;
          else if((typeNode == "LOGICAL") ||
                  (typeNode == "CHARACTER"))
            bytes = 1;
        }

        if(leafPos < expRoot.leafCount){
          if(expRoot[leafPos].value == "*"){
            ++leafPos;
            bytes    = atoi(expRoot[leafPos].value.c_str());
            ++leafPos;
          }
          else if((expRoot[leafPos].value == "(") &&
                  (expRoot[leafPos].leafCount)){

            bytes = atoi(expRoot[leafPos][0].value.c_str());
            ++leafPos;
          }
        }

        switch(bytes){
        case 1:
          typeNode = "char" + suffix; break;
        case 2:
          typeNode = "short" + suffix; break;
        case 4:
          if(isFloat)
            typeNode = "float" + suffix;
          else
            typeNode = "int" + suffix;
          break;
        case 8:
          if(isFloat)
            typeNode = "double" + suffix;
          else
            typeNode = "long long" + suffix;
          break;
        default:
          OCCA_CHECK(false,
                     "Error loading " << typeNode << "(" << bytes << ")");
        };
      }

      return typeNode;
    }

    int varInfo::loadStackPointersFromFortran(expNode &expRoot,
                                              int leafPos){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      if((expRoot[leafPos].value != "(") ||
         (expRoot[leafPos].leafCount == 0)){

        if(expRoot[leafPos].value == "(")
          return (leafPos + 1);

        return leafPos;
      }

      expRoot[leafPos].organize(parsingFortran);

      expNode *expPos = &(expRoot[leafPos][0]);

      bool hasColon = false;

      // rightQualifiers are copied from [firstVar]
      if(rightQualifiers.has("*"))
        rightQualifiers.remove("*");

      if(expPos->value != ","){
        if(expPos->value == ":"){
          pointerCount = 1;
          rightQualifiers.add("*", 0);
        }
        else{
          stackPointerCount = 1;

          stackExpRoots = new expNode[1];

          setupFortranStackExp(stackExpRoots[0],
                               expRoot[leafPos][0]);
        }
      }
      else if((expPos->leafCount) &&
              (expPos->value == ",")){

        stackPointerCount = 1;
        int found = 0;

        for(int pass = 0; pass < 2; ++pass){

          while((expPos->leafCount) &&
                (expPos->value == ",")){

            if(!hasColon)
              hasColon = ((expPos->leaves[0]->value == ":") ||
                          (expPos->leaves[1]->value == ":"));

            if(pass == 0) {
              ++stackPointerCount;
            }
            else {
              setupFortranStackExp(stackExpRoots[found++],
                                   *(expPos->leaves[1]));
            }

            expPos = expPos->leaves[0];
          }

          if(hasColon){
            pointerCount      = stackPointerCount;
            stackPointerCount = 0;

            for(int i = 0; i < pointerCount; ++i)
              rightQualifiers.add("*", 0);

            break;
          }

          if(pass == 0){
            stackExpRoots = new expNode[stackPointerCount];
          }
          else{
            setupFortranStackExp(stackExpRoots[found],
                                 *expPos);
          }

          expPos = &(expRoot[leafPos][0]);
        }
      }

      ++leafPos;

      if(pointerCount &&
         rightQualifiers.has("&")){

        rightQualifiers.remove("&");
      }

      stackPointersUsed = stackPointerCount;

      return leafPos;
    }

    void varInfo::setupFortranStackExp(expNode &stackExp,
                                       expNode &valueExp){
      stackExp.info  = expType::C;
      stackExp.value = "[";

      stackExp.leaves    = new expNode*[1];
      stackExp.leafCount = 1;

      stackExp.leaves[0] = &valueExp;
    }
    //   =====================
    //==================================


    //---[ Variable Info ]------------
    int varInfo::leftQualifierCount(){
      return leftQualifiers.qualifierCount;
    }

    int varInfo::rightQualifierCount(){
      return rightQualifiers.qualifierCount;
    }

    bool varInfo::hasQualifier(const std::string &qName){
      return leftQualifiers.has(qName);
    }

    bool varInfo::hasRightQualifier(const std::string &qName){
      return rightQualifiers.has(qName);
    }

    void varInfo::addQualifier(const std::string &qName, int pos){
      leftQualifiers.add(qName, pos);
    }

    void varInfo::addRightQualifier(const std::string &qName, int pos){
      rightQualifiers.add(qName, pos);
    }

    void varInfo::removeQualifier(const std::string &qName){
      leftQualifiers.remove(qName);
    }

    void varInfo::removeRightQualifier(const std::string &qName){
      rightQualifiers.remove(qName);
    }

    std::string& varInfo::getLeftQualifier(const int pos){
      return leftQualifiers.get(pos);
    }

    std::string& varInfo::getRightQualifier(const int pos){
      return rightQualifiers.get(pos);
    }

    std::string& varInfo::getLastLeftQualifier(){
      return leftQualifiers.get(leftQualifiers.qualifierCount - 1);
    }

    std::string& varInfo::getLastRightQualifier(){
      return rightQualifiers.get(rightQualifiers.qualifierCount - 1);
    }

    expNode& varInfo::stackSizeExpNode(const int pos){
      return stackExpRoots[pos][0];
    }

    void varInfo::removeStackPointers(){
      if(stackPointerCount){
        stackPointerCount = 0;
        stackPointersUsed = 0;

        delete [] stackExpRoots;
        stackExpRoots = NULL;
      }
    }

    varInfo& varInfo::getArgument(const int pos){
      return *(argumentVarInfos[pos]);
    }

    void varInfo::setArgument(const int pos, varInfo &var){
      argumentVarInfos[pos] = &var;
    }

    void varInfo::addArgument(const int pos, varInfo &arg){
      varInfo **newArgumentVarInfos = new varInfo*[argumentCount + 1];

      for(int i = 0; i < pos; ++i)
        newArgumentVarInfos[i] = argumentVarInfos[i];

      newArgumentVarInfos[pos] = &arg;

      for(int i = pos; i < argumentCount; ++i)
        newArgumentVarInfos[i + 1] = argumentVarInfos[i];

      if(argumentCount)
        delete [] argumentVarInfos;

      argumentVarInfos = newArgumentVarInfos;
      ++argumentCount;
    }
    //================================

    bool varInfo::isConst(){
      const int qCount = leftQualifiers.qualifierCount;

      for(int i = 0; i < qCount; ++i){
        const std::string &q = leftQualifiers[i];

        if((q == "const") ||
           (q == "occaConst")){

          return true;
        }
      }

      return false;
    }

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

      if(stackPointerCount && stackPointersUsed){
        if(stackPointersUsed == stackPointerCount){
          for(int i = 0; i < stackPointerCount; ++i)
            ret += (std::string) stackExpRoots[i];
        }
        else{
          ret += "[(";
          ret += (std::string) stackSizeExpNode(0);
          ret += ')';

          for(int i = 1; i < stackPointerCount; ++i){
            ret += "*(";
            ret += (std::string) stackSizeExpNode(i);
            ret += ")";
          }

          ret += ']';
        }
      }

      for(int i = (functionNestCount - 1); 0 <= i; --i){
        ret += functionNests[i].toString();
        ret += ')';
      }

      if(info & varType::functionType){
        ret += '(';

        if(argumentCount){
          ret += argumentVarInfos[0]->toString();

          for(int i = 1; i < argumentCount; ++i){
            ret += ", ";
            ret += argumentVarInfos[i]->toString();
          }
        }

        ret += ')';
      }

      return ret;
    }

    varInfo::operator std::string () {
      return toString();
    }

    std::ostream& operator << (std::ostream &out, varInfo &var){
      out << var.toString();
      return out;
    }
    //============================================


    //---[ Var Dependency Graph ]-----------------
    sDep_t::sDep_t() :
      sID(-1) {}

    sDep_t::sDep_t(const sDep_t &sd){
      *this = sd;
    }

    sDep_t& sDep_t::operator = (const sDep_t &sd){
      sID  = sd.sID;
      deps = sd.deps;

      return *this;
    }

    varInfo& sDep_t::operator [] (const int pos){
      return *(deps[pos]);
    }

    int sDep_t::size(){
      return deps.size();
    }

    void sDep_t::add(varInfo &var){
      deps.push_back(&var);
    }

    void sDep_t::uniqueAdd(varInfo &var){
      if(has(var))
        return;

      deps.push_back(&var);
    }

    bool sDep_t::has(varInfo &var){
      const int depCount = deps.size();

      for(int i = 0; i < depCount; ++i){
        if(deps[i] == &var)
          return true;
      }

      return false;
    }

    varDepGraph::varDepGraph(){}

    varDepGraph::varDepGraph(varInfo &var,
                             statement &sBound){
      setup(var, sBound);
    }

    varDepGraph::varDepGraph(varInfo &var,
                             statement &sBound,
                             statementIdMap_t &idMap){
      setup(var, sBound, idMap);
    }

    varDepGraph::varDepGraph(const varDepGraph &vdg){
      *this = vdg;
    }

    varDepGraph& varDepGraph::operator = (const varDepGraph &vdg){
      sUpdates = vdg.sUpdates;

      return *this;
    }

    void varDepGraph::setup(varInfo &var,
                            statement &sBound){
      statement *globalScope = sBound.getGlobalScope();

      statementIdMap_t idMap;

      globalScope->setStatementIdMap(idMap);

      setup(var, sBound, idMap);
    }

    void varDepGraph::setup(varInfo &var,
                            statement &sBound,
                            statementIdMap_t &idMap){
      statementNode *originSN = &(sBound.varUpdateMap[&var]);
      statementNode *sn       = lastNode(originSN);

      const int sID = idMap[&sBound];

      // Always add the origin statement
      checkStatementForDependency(var,
                                  *(originSN->value),
                                  sID,
                                  idMap);

      while(sn){
        statement &s2 = *(sn->value);

        if(checkStatementForDependency(var, s2, sID, idMap)){
          if(s2.setsVariableValue(var))
            return;
        }
        else
          break;

        sn = sn->left;
      }
    }

    bool varDepGraph::checkStatementForDependency(varInfo &var,
                                                  statement &s,
                                                  const int sBoundID,
                                                  statementIdMap_t &idMap){
      const bool keepGoing = true;  // Just here for readability
      const bool stop      = false;

      if((idMap.find(&s) == idMap.end()) ||  // Skip if statement is not in the map
         (s.info & smntType::functionStatement)){ // Functions don't have dependencies

        return keepGoing;
      }

      const int sID2 = idMap[&s];

      if((sID2 < sBoundID) && !has(sID2)){
        sUpdates.push_back(sDep_t());
        sDep_t &sd = sUpdates.back();

        sd.sID = sID2;
        s.setVariableDeps(var, sd);

        return keepGoing;
      }

      return stop;
    }

    bool varDepGraph::has(const int sID){
      const int updates = sUpdates.size();

      for(int i = 0; i < updates; ++i){
        if(sUpdates[i].sID == sID)
          return true;
      }

      return false;
    }

    void varDepGraph::addDependencyMap(idDepMap_t &depMap){
      const int updates = sUpdates.size();

      if(updates == 0)
        return;

      for(int i = 0; i < updates; ++i){
        if(0 <= sUpdates[i].sID)
          depMap[sUpdates[i].sID] = true;
      }
    }

    void varDepGraph::addFullDependencyMap(idDepMap_t &depMap,
                                           statementIdMap_t &idMap){
      statementVector_t sVec;
      statement::setStatementVector(idMap, sVec);

      addFullDependencyMap(depMap, idMap, sVec);
    }

    // [-] Missing nested loop checks
    //    Example: Dependent variable is inside a flow statement
    void varDepGraph::addFullDependencyMap(idDepMap_t &depMap,
                                           statementIdMap_t &idMap,
                                           statementVector_t &sVec){

      const int updates = sUpdates.size();

      if(updates == 0)
        return;

      for(int i = 0; i < updates; ++i){
        sDep_t &sDep = sUpdates[i];

        if(depMap.find(sDep.sID) != depMap.end())
          continue;

        depMap[sDep.sID] = true;

        const int varCount = sDep.size();
        statement &s = *(sVec[sDep.sID]);

        for(int v = 0; v < varCount; ++v){
          varInfo &var = sDep[v];

          if(&var != NULL){
            varDepGraph vdg(sDep[v], s, idMap);

            vdg.addFullDependencyMap(depMap, idMap, sVec);
          }
        }
      }
    }
    //============================================


    //---[ Kernel Info ]--------------------------
    argumentInfo::argumentInfo() :
      pos(0),
      isConst(false) {}

    argumentInfo::argumentInfo(const argumentInfo &info) :
      pos(info.pos),
      isConst(info.isConst) {}

    argumentInfo& argumentInfo::operator = (const argumentInfo &info){
      pos     = info.pos;
      isConst = info.isConst;

      return *this;
    }

    kernelInfo::kernelInfo() :
      name(),
      baseName() {}

    kernelInfo::kernelInfo(const kernelInfo &info) :
      name(info.name),
      baseName(info.baseName),
      nestedKernels(info.nestedKernels),
      argumentInfos(info.argumentInfos) {}

    kernelInfo& kernelInfo::operator = (const kernelInfo &info){
      name     = info.name;
      baseName = info.baseName;

      nestedKernels = info.nestedKernels;
      argumentInfos = info.argumentInfos;

      return *this;
    }

    occa::parsedKernelInfo kernelInfo::makeParsedKernelInfo(){
      occa::parsedKernelInfo kInfo;

      kInfo.name     = name;
      kInfo.baseName = baseName;

      kInfo.nestedKernels = nestedKernels.size();

      kInfo.argumentInfos = argumentInfos;

      return kInfo;
    }
    //==============================================
  };

  //---[ Parsed Kernel Info ]---------------------
  parsedKernelInfo::parsedKernelInfo() :
    name(""),
    baseName(""),
    nestedKernels(0) {}

  parsedKernelInfo::parsedKernelInfo(const parsedKernelInfo &kInfo) :
    name(kInfo.name),
    baseName(kInfo.baseName),
    nestedKernels(kInfo.nestedKernels),
    argumentInfos(kInfo.argumentInfos) {}

  parsedKernelInfo& parsedKernelInfo::operator = (const parsedKernelInfo &kInfo){
    name     = kInfo.name;
    baseName = kInfo.baseName;

    nestedKernels = kInfo.nestedKernels;

    argumentInfos = kInfo.argumentInfos;

    return *this;
  }

  void parsedKernelInfo::removeArg(const int pos){
    argumentInfos.erase(argumentInfos.begin() + pos);
  }
  //==============================================
};
