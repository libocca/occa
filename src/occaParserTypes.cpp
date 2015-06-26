#include "occaParserTypes.hpp"
#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
    bool isInlinedASM(const std::string &attrName){
      return (attrName == "__asm");
    }

    bool isInlinedASM(expNode &expRoot, int leafPos){
      if(expRoot.leafCount <= leafPos)
        return false;

      return isInlinedASM(expRoot[leafPos].value);
    }

    bool isAnAttribute(const std::string &attrName){
      return ((attrName == "@") ||
              (attrName == "__attribute__"));
    }

    bool isAnAttribute(expNode &expRoot, int leafPos){
      if(expRoot.leafCount <= leafPos)
        return false;

      return isAnAttribute(expRoot[leafPos].value);
    }

    int skipAttribute(expNode &expRoot, int leafPos){
      if(!isAnAttribute(expRoot, leafPos))
        return leafPos;

      const std::string &attrTag = expRoot[leafPos].value;
      ++leafPos;

      if((attrTag == "@")              &&
         (leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value != "(")){

        ++leafPos;
      }

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "(")){

        ++leafPos;
      }

      return leafPos;
    }

    //---[ Attribute Class ]----------------------
    attribute_t::attribute_t() :
      argCount(0),
      args(NULL),

      value(NULL) {}

    attribute_t::attribute_t(expNode &e) :
      argCount(0),
      args(NULL),

      value(NULL) {

      load(e);
    }

    attribute_t::attribute_t(const attribute_t &attr) :
      name(attr.name),

      argCount(attr.argCount),
      args(attr.args),

      value(attr.value) {}

    attribute_t& attribute_t::operator = (const attribute_t &attr){
      name = attr.name;

      argCount = attr.argCount;
      args     = attr.args;

      value = attr.value;

      return *this;
    }

    void attribute_t::load(expNode &e){
      const int attrIsSet = (e.value == "=");

      expNode &attrNode = (attrIsSet ? e[0] : e);

      if(startsSection(attrNode.value))
        loadVariable(attrNode);
      else
        name = attrNode.value;

      if(!attrIsSet)
        return;

      value = e[1].clonePtr();
    }

    void attribute_t::loadVariable(expNode &e){
      name = e[0].value;

      expNode &csvFlatRoot = *(e[1].makeCsvFlatHandle());

      argCount = csvFlatRoot.leafCount;

      if(argCount){
        args = new expNode*[argCount];

        for(int i = 0; i < argCount; ++i)
          args[i] = csvFlatRoot[i].clonePtr();
      }

      expNode::freeFlatHandle(csvFlatRoot);
    }

    expNode& attribute_t::operator [] (const int pos){
      return *(args[pos]);
    }

    std::string attribute_t::argStr(const int pos){
      return args[pos]->toString();
    }

    expNode& attribute_t::valueExp(){
      return *value;
    }

    std::string attribute_t::valueStr(){
      return value->toString();
    }

    attribute_t::operator std::string(){
      std::string ret;

      ret += name;

      if(argCount){
        ret += "(";

        for(int i = 0; i < argCount; ++i){
          if(i)
            ret += ", ";

          ret += argStr(i);
        }

        ret += ")";
      }

      if(value){
        ret += " = ";
        ret += valueStr();
      }

      return ret;
    }

    std::ostream& operator << (std::ostream &out, attribute_t &attr){
      out << (std::string) attr;

      return out;
    }

    void setAttributeMap(attributeMap_t &attributeMap,
                         const std::string &attrName){

      attribute_t &attr = *(new attribute_t());
      attr.name = attrName;

      attributeMap[attrName] = &attr;
    }

    int setAttributeMap(attributeMap_t &attributeMap,
                        expNode &expRoot,
                        int leafPos){

      while(true){
        const int leafPos2 = leafPos;

        leafPos = setAttributeMapR(attributeMap, expRoot, leafPos);

        if(leafPos == leafPos2)
          break;
      }

      return leafPos;
    }

    int setAttributeMapR(attributeMap_t &attributeMap,
                         expNode &expRoot,
                         int leafPos){

      if(!isAnAttribute(expRoot, leafPos))
        return leafPos;

      if(expRoot[leafPos].value == "__attribute__"){
        ++leafPos;

        if(leafPos < expRoot.leafCount){
          expNode &tmp = *(expRoot[leafPos].clonePtr());
          tmp.changeExpTypes();
          tmp.organize();

          attribute_t &attr = *(new attribute_t());
          attr.name = tmp.toString();

          attributeMap["__attribute__"] = &attr;

          tmp.free();

          ++leafPos;
        }

        return leafPos;
      }

      ++leafPos;

      // Only one attribute
      if((expRoot[leafPos].info & expType::C) == 0){
        expNode attrNode;

        expRoot[leafPos].info |= expType::attribute;

        const int leafStart = leafPos;

        if(((leafPos + 1) < expRoot.leafCount) &&
           (expRoot[leafPos + 1].info & expType::C)){

          ++leafPos;
        }

        ++leafPos;

        attrNode.copyAndUseExpLeaves(expRoot,
                                     leafStart, (leafPos - leafStart));
        attrNode.organize();

        attribute_t &attr = *(new attribute_t(attrNode[0]));

        attributeMap[attr.name] = &attr;

        attrNode.free();

        return leafPos;
      }
      else {
        expNode attrRoot = expRoot[leafPos].clone();

        for(int i = 0; i < attrRoot.leafCount; ++i){
          if(attrRoot[i].info & expType::unknown)
            attrRoot[i].info |= expType::attribute;
        }

        attrRoot.organize();

        expNode &csvFlatRoot = *(attrRoot[0].makeCsvFlatHandle());

        const int attributeCount = csvFlatRoot.leafCount;

        for(int i = 0; i < attributeCount; ++i){
          expNode &attrNode = csvFlatRoot[i];

          attribute_t &attr = *(new attribute_t(attrNode));

          attributeMap[attr.name] = &attr;
        }

        attrRoot.free();
        expNode::freeFlatHandle(csvFlatRoot);
      }

      return (leafPos + 1);
    }

    void printAttributeMap(attributeMap_t &attributeMap){
      std::cout << attributeMapToString(attributeMap) << '\n';
    }

    std::string attributeMapToString(attributeMap_t &attributeMap){
      std::string ret;

      if(attributeMap.size()){
        attributeMapIterator it = attributeMap.begin();

        const bool putParentheses = ((1 < attributeMap.size()) ||
                                     (it->second->value != NULL));

        ret += '@';

        if(putParentheses)
          ret += '(';

        bool oneAttrSet = false;

        while(it != attributeMap.end()){
          if(oneAttrSet)
            ret += ", ";
          else
            oneAttrSet = true;

          ret += (std::string) *(it->second);

          ++it;
        }

        if(putParentheses)
          ret += ')';
      }

      return ret;
    }
    //==================================


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

      OCCA_CHECK(expRoot.sInfo != NULL,
                 "Cannot load [qualifierInfo] without a statement");

      varInfo var;

      return loadFrom(*(expRoot.sInfo), var, expRoot, leafPos);
    }

    int qualifierInfo::loadFrom(varInfo &var,
                                expNode &expRoot,
                                int leafPos){

      OCCA_CHECK(expRoot.sInfo != NULL,
                 "Cannot load [qualifierInfo] without a statement");

      return loadFrom(*(expRoot.sInfo), var, expRoot, leafPos);
    }

    int qualifierInfo::loadFrom(statement &s,
                                expNode &expRoot,
                                int leafPos){

      varInfo var;

      return loadFrom(s, var, expRoot, leafPos);
    }

    int qualifierInfo::loadFrom(statement &s,
                                varInfo &var,
                                expNode &expRoot,
                                int leafPos){

      if(expRoot.leafCount <= leafPos)
        return leafPos;

      qualifierCount = 0;

      const int leafRoot = leafPos;

      for(int pass = 0; pass < 2; ++pass){
        if(pass == 1){
          if(qualifierCount)
            qualifiers = new std::string[qualifierCount];
          else
            break;
        }

        qualifierCount = 0;
        leafPos = leafRoot;

        while(leafPos < expRoot.leafCount){
          if(expHasQualifier(expRoot, leafPos)){
            if(pass == 1){
              if((expRoot[leafPos].value == "*") &&
                 hasImplicitInt()){
                break;
              }

              qualifiers[qualifierCount] = expRoot[leafPos].value;
            }

            ++qualifierCount;
            ++leafPos;
          }
          else if(isAnAttribute(expRoot, leafPos)){
            const bool is__attribute__ = (expRoot[leafPos].value == "__attribute__");

            if(pass == 0){
              leafPos = skipAttribute(expRoot, leafPos);
            }
            else {
              leafPos = setAttributeMap(var.attributeMap, expRoot, leafPos);

              if(is__attribute__){
                attributeMapIterator it = var.attributeMap.find("__attribute__");
                attribute_t &attr       = *(it->second);

                qualifiers[qualifierCount] = ("__attribute__" + attr.name);

                var.attributeMap.erase(it);
              }
            }

            if(is__attribute__)
              ++qualifierCount;
          }
          else
            break;
        }
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
        add("occaConst", 0);
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
      if(fortranVarNeedsUpdate(var, expPos[leafPos].value))
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
              add("occaConst", 0);

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
    int qualifierInfo::has(const std::string &qName){
      int count = 0;

      for(int i = 0; i < qualifierCount; ++i){
        if(qualifiers[i] == qName)
          ++count;
      }

      return count;
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

    bool qualifierInfo::hasImplicitInt(){
      return (has("unsigned") ||
              has("signed")   ||
              has("short")    ||
              has("long"));
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

    bool expHasQualifier(expNode &allExp, int expPos){
      return (allExp[expPos].info & expType::qualifier);
    }
    //============================================


    //---[ Type Info Class ]----------------------
    typeInfo::typeInfo() :
      leftQualifiers(),

      name(""),

      thType(noType),

      nestedInfoCount(0),
      nestedExps(NULL),

      typedefHasDefinition(false),
      typedefing(NULL),
      baseType(NULL),

      typedefVar(NULL) {}

    typeInfo::typeInfo(const typeInfo &type) :
      leftQualifiers(type.leftQualifiers),

      name(type.name),

      thType(type.thType),

      nestedInfoCount(type.nestedInfoCount),
      nestedExps(type.nestedExps),

      typedefHasDefinition(type.typedefHasDefinition),
      typedefing(type.typedefing),
      baseType(type.baseType),

      typedefVar(type.typedefVar),

      opOverloadMaps(type.opOverloadMaps) {}

    typeInfo& typeInfo::operator = (const typeInfo &type){
      leftQualifiers = type.leftQualifiers;

      name = type.name;

      thType = type.thType;

      nestedInfoCount = type.nestedInfoCount;
      nestedExps      = type.nestedExps;

      typedefHasDefinition = type.typedefHasDefinition;
      typedefing           = type.typedefing;
      baseType             = type.baseType;

      typedefVar = type.typedefVar;

      opOverloadMaps = type.opOverloadMaps;

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
                           int leafPos,
                           bool addTypeToScope){

      OCCA_CHECK(expRoot.sInfo != NULL,
                 "Cannot load [typeInfo] without a statement");

      return loadFrom(*(expRoot.sInfo),
                      expRoot,
                      leafPos,
                      addTypeToScope);
    }

    int typeInfo::loadFrom(statement &s,
                           expNode &expRoot,
                           int leafPos,
                           bool addTypeToScope){

      if(expRoot.leafCount <= leafPos)
        return leafPos;

      // Typedefs are pre-loaded with qualifiers
      if(leftQualifiers.size() == 0)
        leafPos = leftQualifiers.loadFrom(s, expRoot, leafPos);

      if(leftQualifiers.has("typedef")){
        leafPos = loadTypedefFrom(s, expRoot, leafPos);

        if(addTypeToScope &&
           (s.up != NULL)){

          s.up->addType(*this);
        }

        return leafPos;
      }

      baseType = this;

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].info & expType::unknown)){

        name = expRoot[leafPos++].value;

        if(addTypeToScope &&
           (s.up != NULL)){

          s.up->addType(*this);
        }

        updateThType();
      }
      else if(hasImplicitInt()){
        name     = "int";
        baseType = s.hasTypeInScope("int");
      }

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "{")){

        expNode &leaf = expRoot[leafPos++];

        if(leftQualifiers.has("enum")){
          nestedInfoCount = 1;
          nestedExps      = new expNode[nestedInfoCount];

          nestedExps[0] = leaf.clone();
          nestedExps[0].organizeDeclareStatement(expFlag::none);

          return leafPos;
        }

        nestedInfoCount = delimiterCount(leaf, ";");
        nestedExps      = new expNode[nestedInfoCount];

        int sLeafPos = 0;

        for(int i = 0; i < nestedInfoCount; ++i){
          int sNextLeafPos = nextDelimiter(leaf, sLeafPos, ";");

          // Empty statements
          if(sNextLeafPos != sLeafPos){
            const bool loadType = typeInfo::statementIsATypeInfo(s,
                                                                 leaf,
                                                                 sLeafPos);

            sNextLeafPos = leaf.mergeRange(expType::root,
                                           sLeafPos,
                                           sNextLeafPos);

            expNode::swap(nestedExps[i], leaf[sLeafPos]);

            // For nested types, types are still
            //   labeled as expType::unknown
            nestedExps[i].labelReferenceQualifiers();

            if(!loadType)
              nestedExps[i].organizeDeclareStatement(expFlag::none);
            else
              nestedExps[i].organizeStructStatement();

            leaf.setLeaf(nestedExps[i], sLeafPos);
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

        typeInfo *leafType = s.hasTypeInScope(expRoot[leafPos].value);

        if(leafType){
          typedefing = leafType;

          ++leafPos;
        }
        else{
          if(!leftQualifiers.hasImplicitInt()){
            typedefing           = new typeInfo;
            typedefing->name     = expRoot[leafPos].value;
            typedefing->baseType = typedefing;

            ++leafPos;
          }
          else {
            typedefing = s.hasTypeInScope("int");
          }
        }
      }

      if((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "{")){

        // Anonymous type
        if(typedefing == NULL){
          typedefing           = new typeInfo;
          typedefing->baseType = typedefing;

          std::string transferQuals[4] = {"class", "enum", "union", "struct"};

          for(int i = 0; i < 4; ++i){
            if(leftQualifiers.has(transferQuals[i])){
              leftQualifiers.remove(transferQuals[i]);
              typedefing->leftQualifiers.add(transferQuals[i]);
            }
          }
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

      updateThType();

      return leafPos;
    }

    void typeInfo::updateThType(){
      if(name == "bool")
        thType = boolType;
      else if(name == "char")
        thType = charType;
      else if(name == "float")
        thType = floatType;
      else if(name == "double")
        thType = doubleType;
      else {
        if(name == "short"){
          const bool unsigned_ = hasQualifier("unsigned");

          thType = (unsigned_ ? ushortType : shortType);
        }
        else if((name == "int") ||
                (name == "long")){

          const bool unsigned_ = hasQualifier("unsigned");
          const int longs_     = hasQualifier("long");

          switch(longs_){
          case 0:
            thType = (unsigned_ ? uintType      : intType);
          case 1:
            thType = (unsigned_ ? ulongType     : longType);
          default:
            thType = (unsigned_ ? ulonglongType : longlongType);
          }
        }
        else
          thType = noType;
      }
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

      if(qualifiers.hasImplicitInt())
        return false;

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
    int typeInfo::hasQualifier(const std::string &qName){
      return leftQualifiers.has(qName);
    }

    void typeInfo::addQualifier(const std::string &qName,
                                int pos){
      leftQualifiers.add(qName, pos);
    }

    bool typeInfo::hasImplicitInt(){
      return leftQualifiers.hasImplicitInt();
    }

    int typeInfo::pointerDepth(){
      if(typedefing)
        return typedefVar->pointerDepth();

      return 0;
    }
    //==================================


    //---[ Class Info ]---------------
    varInfo* typeInfo::hasOperator(const std::string &name){
      return NULL;
    }
    //================================

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
        const bool isAnEnum = leftQualifiers.has("enum");

        ret += tab;
        ret += leftQualifiers.toString();
        ret += name;

        if(nestedInfoCount){
          if(name.size())
            ret += ' ';

          ret += '{';
          ret += '\n';

          for(int i = 0; i < nestedInfoCount; ++i){
            if(!isAnEnum){
              ret += nestedExps[i].toString(tab + "  ");
            }
            else {
              if(i < (nestedInfoCount - 1)){
                ret += nestedExps[i].toString(tab + "  ", (expFlag::noSemicolon |
                                                           expFlag::endWithComma));
              }
              else {
                ret += nestedExps[i].toString(tab + "  ", expFlag::noSemicolon);
              }
            }

            if(back(ret) != '\n')
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
      stackPointersUsed(0),
      stackExpRoots(NULL),

      bitfieldSize(-1),

      usesTemplate(false),
      tArgCount(0),
      tArgs(NULL),

      argumentCount(0),
      argumentVarInfos(NULL),

      functionNestCount(0),
      functionNests(NULL) {}

    varInfo::varInfo(const varInfo &var) :
      info(var.info),

      attributeMap(var.attributeMap),
      leftQualifiers(var.leftQualifiers),
      rightQualifiers(var.rightQualifiers),

      baseType(var.baseType),

      name(var.name),

      pointerCount(var.pointerCount),

      stackPointerCount(var.stackPointerCount),
      stackPointersUsed(var.stackPointersUsed),
      stackExpRoots(var.stackExpRoots),

      bitfieldSize(var.bitfieldSize),

      dimAttr(var.dimAttr),
      idxOrdering(var.idxOrdering),

      usesTemplate(var.usesTemplate),
      tArgCount(var.tArgCount),
      tArgs(var.tArgs),

      argumentCount(var.argumentCount),
      argumentVarInfos(var.argumentVarInfos),

      functionNestCount(var.functionNestCount),
      functionNests(var.functionNests) {}

    varInfo& varInfo::operator = (const varInfo &var){
      info = var.info;

      attributeMap    = var.attributeMap;
      leftQualifiers  = var.leftQualifiers;
      rightQualifiers = var.rightQualifiers;

      baseType = var.baseType;

      name = var.name;

      pointerCount = var.pointerCount;

      stackPointerCount  = var.stackPointerCount;
      stackPointersUsed  = var.stackPointersUsed;
      stackExpRoots      = var.stackExpRoots;

      bitfieldSize = var.bitfieldSize;

      dimAttr     = var.dimAttr;
      idxOrdering = var.idxOrdering;

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

      v.attributeMap    = attributeMap;
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

    varInfo* varInfo::clonePtr(){
      return new varInfo(clone());
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

      OCCA_CHECK(expRoot.sInfo != NULL,
                 "Cannot load [varInfo] without a statement");

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

      if((leafPos < expRoot.leafCount) &&
         expRoot[leafPos].value == ":"){

        ++leafPos;

        if(leafPos < expRoot.leafCount){
          typeHolder th(expRoot[leafPos].value);

          OCCA_CHECK(th.type != noType,
                     "Bitfield is not known at compile-time");

          bitfieldSize = th.to<int>();

          ++leafPos;
        }
      }

      leafPos = setAttributeMap(attributeMap, expRoot, leafPos);

      setupAttributes();

      organizeExpNodes();

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
        else if(leftQualifiers.has("unsigned"))
          baseType = s.hasTypeInScope("int");
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

      if(expRoot.leafCount <= leafPos)
        return varType::var;

      if(expRoot[leafPos].value == "("){
        ++leafPos;

        if(expRoot.leafCount <= leafPos)
          return varType::functionDec;

        if((expRoot[leafPos].value == "{") ||
           isInlinedASM(expRoot, leafPos)){

          return varType::functionDef;
        }

        return varType::functionDec;
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

        if((leaf->leaves[0]->value == "*") ||
           (leaf->leaves[0]->value == "^")){

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

      if(nodeHasName(expRoot, leafPos))
        return loadNameFromNode(expRoot, leafPos);

      expNode *expRoot2 = &expRoot;
      int leafPos2      = leafPos;
      expNode *leaf     = expRoot2->leaves[leafPos2];

      int nestPos = 0;

      while((leaf != NULL)            &&
            (leaf->info & expType::C) &&
            (0 < leaf->leafCount)     &&
            (leaf->value == "(")){

        if(((*leaf)[0].value == "*") ||
           ((*leaf)[0].value == "^")){

          if((*leaf)[0].value == "^")
            info |= varType::block;

          if((leafPos2 + 1) < (expRoot2->leafCount)){
            leaf = expRoot2->leaves[leafPos2 + 1];

            if((leaf->info & expType::C) &&
               (leaf->value == "(")){

              functionNests[nestPos].info = varType::function;
              functionNests[nestPos].loadArgsFrom(s, *expRoot2, leafPos2 + 1);
            }
          }

          expRoot2 = expRoot2->leaves[leafPos2];
          leafPos2 = 1;

          leaf = ((leafPos2 < expRoot.leafCount) ?
                  expRoot2->leaves[leafPos2] :
                  NULL);

          ++nestPos;
        }
        else {
          break;
        }
      }

      if((expRoot2 != &expRoot) &&
         (nodeHasName(*expRoot2, leafPos2))){

        leafPos2 = loadNameFromNode(*expRoot2, leafPos2);

        if((leafPos2 < expRoot2->leafCount) &&
           expRoot2->leaves[leafPos2]->value == "("){

          info = varType::function;
          loadArgsFrom(s, *expRoot2, leafPos2);
        }

        // Skip the name and function-pointer arguments
        leafPos += 2;
      }

      return leafPos;
    }

    bool varInfo::nodeHasName(expNode &expRoot,
                              int leafPos){
      if(expRoot.leafCount <= leafPos)
        return false;

      return (expRoot[leafPos].info & (expType::unknown  |
                                       expType::varInfo  |
                                       expType::function));
    }

    int varInfo::loadNameFromNode(expNode &expRoot,
                                  int leafPos){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      expNode *leaf = expRoot.leaves[leafPos];

      if(leaf->info & (expType::unknown  |
                       expType::varInfo  |
                       expType::function)){

        if(leaf->info & expType::varInfo)
          name = leaf->getVarInfo().name;
        else
          name = leaf->value;

        return loadStackPointersFrom(expRoot, leafPos + 1);
      }

      return leafPos;
    }

    int varInfo::loadStackPointersFrom(expNode &expRoot,
                                       int leafPos){
      if(expRoot.leafCount <= leafPos)
        return leafPos;

      stackPointerCount = 0;

      for(int i = leafPos; i < expRoot.leafCount; ++i){
        if(expRoot[i].value == "[")
          ++stackPointerCount;
        else
          break;
      }

      if(stackPointerCount){
        stackExpRoots = new expNode[stackPointerCount];

        for(int i = 0; i < stackPointerCount; ++i){
          if(expRoot[leafPos + i].leafCount){
            expRoot[leafPos + i].cloneTo(stackExpRoots[i]);

            stackExpRoots[i].info  = expType::root;
            stackExpRoots[i].value = "";
          }
        }
      }

      stackPointersUsed = stackPointerCount;

      return (leafPos + stackPointerCount);
    }

    int varInfo::loadArgsFrom(statement &s,
                              expNode &expRoot,
                              int leafPos){
      if( !(info & varType::function) )
        return leafPos;

      OCCA_CHECK(leafPos < expRoot.leafCount,
                 "Missing arguments from function variable");

      if(expRoot[leafPos].leafCount){
        expNode &leaf = expRoot[leafPos];
        int sLeafPos  = 0;

        argumentCount    = 1 + typeInfo::delimiterCount(leaf, ",");
        argumentVarInfos = new varInfo*[argumentCount];

        for(int i = 0; i < argumentCount; ++i){
          if(leaf[sLeafPos].value == "..."){
            OCCA_CHECK(i == (argumentCount - 1),
                       "Variadic argument [...] has to be the last argument");

            info |= varType::variadic;

            --argumentCount;
            break;
          }


          argumentVarInfos[i] = new varInfo();
          sLeafPos = argumentVarInfos[i]->loadFrom(s, leaf, sLeafPos);
          sLeafPos = typeInfo::nextDelimiter(leaf, sLeafPos, ",") + 1;
        }
      }

      return (leafPos + 1);
    }

    void varInfo::setupAttributes(){
      attributeMapIterator it = attributeMap.find("dim");

      if(it != attributeMap.end())
        dimAttr = *(it->second);

      it = attributeMap.find("idxOrder");

      if(it != attributeMap.end()){
        attribute_t &idxOrderAttr = *(it->second);

        OCCA_CHECK(idxOrderAttr.argCount == dimAttr.argCount,
                   "Variable [" << *this << "] has attributes dim(...) and idxOrder(...) with different dimensions");

        const int dims = dimAttr.argCount;

        bool *idxFound = new bool[dims];

        idxOrdering.clear();

        for(int i = 0; i < dims; ++i){
          idxFound[i] = false;
          idxOrdering.push_back(0);
        }

        for(int i = 0; i < dims; ++i){
          typeHolder th;

          bool foundIdx = false;

          if((idxOrderAttr[i].leafCount    == 0) &&
             (idxOrderAttr[i].value.size() == 1)){

            const char c = idxOrderAttr[i].value[0];

            if(('w' <= c) && (c <= 'z')){
              th = (int) (((c - 'w') + 3) % 4); // [w,x,y,z] -> [x,y,z,w]
              foundIdx = true;
            }
            else if(('W' <= c) && (c <= 'Z')){
              th = (int) (((c - 'W') + 3) % 4); // [W,X,Y,Z] -> [X,Y,Z,W]
              foundIdx = true;
            }
          }

          if(!foundIdx){
            OCCA_CHECK(idxOrderAttr[i].valueIsKnown(),
                       "Variable [" << *this << "] has the attribute [" << idxOrderAttr << "] with ordering not known at compile time");

            th = idxOrderAttr[i].calculateValue();

            OCCA_CHECK(!th.isAFloat(),
                       "Variable [" << *this << "] has the attribute [" << idxOrderAttr << "] with a non-integer ordering");
          }

          const int idxOrder = th.to<int>();

          idxOrdering[idxOrder] = i;

          OCCA_CHECK(idxFound[idxOrder] == false,
                     "Variable [" << *this << "] has the attribute [" << idxOrderAttr << "] with a repeating index");

          OCCA_CHECK((0 <= idxOrder) && (idxOrder < dims),
                     "Variable [" << *this << "] has the attribute [" << idxOrderAttr << "] with an index [" << idxOrder << "] outside the range [0," << (dims - 1) << "]");

          idxFound[idxOrder] = true;
        }
      }
    }

    //   ---[ Fortran ]-------
    int varInfo::loadFromFortran(expNode &expRoot,
                                 int leafPos,
                                 varInfo *varHasType){

      OCCA_CHECK(expRoot.sInfo != NULL,
                 "Cannot load [varInfo] without a statement");

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

        if(info & varType::function){
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

      OCCA_CHECK(expRoot.sInfo != NULL,
                 "Cannot load [varInfo] without a statement");

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

        const int typeNodeChars  = typeNode.size();
        const bool typeHasSuffix = isADigit(typeNode[typeNodeChars - 1]);

        std::string suffix = "";

        if(typeHasSuffix){
          for(int i = 0; i < typeNodeChars; ++i){
            if(isADigit(typeNode[i]))
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

      // rightQualifiers are copied from [firstVar]
      if(rightQualifiers.has("*"))
        rightQualifiers.remove("*");

      expRoot[leafPos].changeExpTypes();
      expRoot[leafPos].organize(parserInfo::parsingFortran);

      expNode &csvFlatRoot = *(expRoot[leafPos][0].makeCsvFlatHandle());

      for(int i = 0; i < csvFlatRoot.leafCount; ++i){
        expNode &stackNode = csvFlatRoot[i];

        if(stackNode.value == ":"){
          pointerCount      = csvFlatRoot.leafCount;
          stackPointerCount = 0;

          for(int i = 0; i < pointerCount; ++i)
            rightQualifiers.add("*", 0);

          break;
        }
        else {
          ++stackPointerCount;
        }
      }

      if(stackPointerCount){
        stackExpRoots = new expNode[stackPointerCount];

        for(int i = 0; i < stackPointerCount; ++i){
          stackExpRoots[i].info  = expType::root;
          stackExpRoots[i].value = "";

          stackExpRoots[i].addNode();
          csvFlatRoot[i].cloneTo(stackExpRoots[i][0]);
        }
      }

      expNode::freeFlatHandle(csvFlatRoot);

      ++leafPos;

      if(pointerCount &&
         rightQualifiers.has("&")){

        rightQualifiers.remove("&");
      }

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

    void varInfo::organizeExpNodes(){
      for(int i = 0; i < stackPointerCount; ++i){
        if(!stackExpRoots[i].isOrganized()){
          stackExpRoots[i].changeExpTypes();
          stackExpRoots[i].initOrganization();
          stackExpRoots[i].organize();
        }
      }

      for(int i = 0; i < argumentCount; ++i)
        argumentVarInfos[i]->organizeExpNodes();

      for(int i = 0; i < functionNestCount; ++i)
        functionNests[i].organizeExpNodes();
    }
    //==================================


    //---[ Variable Info ]------------
    attribute_t* varInfo::hasAttribute(const std::string &attr){
      attributeMapIterator it = attributeMap.find(attr);

      if(it == attributeMap.end())
        return NULL;

      return (it->second);
    }

    int varInfo::leftQualifierCount(){
      return leftQualifiers.qualifierCount;
    }

    int varInfo::rightQualifierCount(){
      return rightQualifiers.qualifierCount;
    }

    int varInfo::hasQualifier(const std::string &qName){
      return leftQualifiers.has(qName);
    }

    int varInfo::hasRightQualifier(const std::string &qName){
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

    int varInfo::pointerDepth(){
      if(baseType)
        return (pointerCount + stackPointerCount + baseType->pointerDepth());
      else
        return (pointerCount + stackPointerCount);
    }

    expNode& varInfo::stackSizeExpNode(const int pos){
      return stackExpRoots[pos];
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


    //---[ Class Info ]---------------
    varInfo* varInfo::hasOperator(const std::string &op){
      if(op.size() == 0)
        return NULL;

      if(pointerDepth())
        return (varInfo*) -1; // Dummy non-zero value

      if(baseType)
        return baseType->hasOperator(op);

      return NULL;
    }

    bool varInfo::canBeCastedTo(varInfo &var){
      if(((    baseType->thType & noType) == 0) &&
         ((var.baseType->thType & noType) == 0)){

        return true;
      }

      return false;
    }

    bool varInfo::hasSameTypeAs(varInfo &var){
      if(baseType != var.baseType)
        return false;

      if(stackPointerCount != var.stackPointerCount)
        return false;

      // [-] Need to check if void* is an exception
      if(pointerCount != var.pointerCount)
        return false;

      return true;
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

    void varInfo::printDebugInfo(){
      std::cout << toString(true) << ' ' << attributeMapToString(attributeMap) << '\n';
    }

    std::string varInfo::toString(const bool printType, const std::string &tab){
      std::string ret;

      bool addSpaceBeforeName = false;

      if(printType){
        ret += leftQualifiers.toString();

        if(baseType)
          ret += baseType->name;

        addSpaceBeforeName = !((rightQualifiers.qualifierCount) ||
                               (name.size()));

        if(!addSpaceBeforeName){
          if((info & varType::function)       &&
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

      for(int i = 0; i < (functionNestCount - 1); ++i)
        ret += "(*";

      if(functionNestCount){
        if((info & varType::block) == 0)
          ret += "(*";
        else
          ret += "(^";
      }

      if(addSpaceBeforeName &&
         (name.size() != 0))
        ret += ' ';

      ret += name;

      if(stackPointerCount && stackPointersUsed){
        if(stackPointersUsed == stackPointerCount){
          for(int i = 0; i < stackPointerCount; ++i){
            ret += '[';
            ret += (std::string) stackExpRoots[i];
            ret += ']';
          }
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

      if(info & varType::function){
        ret += '(';

        if(argumentCount){
          ret += argumentVarInfos[0]->toString();

          for(int i = 1; i < argumentCount; ++i){
            ret += ", ";
            ret += argumentVarInfos[i]->toString();
          }
        }

        if(info & varType::variadic){
          if(argumentCount)
            ret += ", ";

          ret += "...";
        }

        ret += ')';
      }

      for(int i = (functionNestCount - 1); 0 <= i; --i){
        ret += ')';
        ret += functionNests[i].toString();
      }

      if(0 <= bitfieldSize){
        ret += " : ";
        ret += occa::toString(bitfieldSize);
      }

      attribute_t *attr = hasAttribute("__attribute__");

      if(attr != NULL){
        ret += " __attribute__";
        ret += attr->name;
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


    //---[ Overloaded Operator Class ]------------
    void overloadedOp_t::add(varInfo &function){
      functions.push_back(&function);
    }

    varInfo* overloadedOp_t::getFromArgs(const int argumentCount,
                                         expNode *arguments){

      varInfo *argumentTypes = new varInfo[argumentCount];

      for(int i = 0; i < argumentCount; ++i)
        argumentTypes[i] = arguments[i].evaluateType();

      varInfo *ret = getFromTypes(argumentCount,
                                  argumentTypes);

      delete [] argumentTypes;

      return ret;
    }

    varInfo* overloadedOp_t::getFromTypes(const int argumentCount,
                                          varInfo *argumentTypes){

      const int functionCount = (int) functions.size();

      varInfoVector_t candidates;

      for(int i = 0; i < functionCount; ++i){
        varInfo &f = argumentTypes[i];
        int arg;

        if(f.argumentCount != argumentCount)
          continue;

        for(arg = 0; arg < argumentCount; ++arg){
          if(!argumentTypes[arg].canBeCastedTo(f.getArgument(arg)))
            break;
        }

        if(arg == argumentCount)
          candidates.push_back(&f);
      }

      return bestFitFor(argumentCount,
                        argumentTypes,
                        candidates);
    }

    varInfo* overloadedOp_t::bestFitFor(const int argumentCount,
                                        varInfo *argumentTypes,
                                        varInfoVector_t &candidates){

      const int candidateCount = (int) candidates.size();

      if(candidateCount == 0)
        return NULL;
      else if(candidateCount == 1)
        return candidates[0];

      int nonAmbiguousCount = candidateCount;
      bool *ambiguous       = new bool[candidateCount];

      for(int i = 0; i < candidateCount; ++i)
        ambiguous[i] = false;

      for(int arg = 0; arg < argumentCount; ++arg){
        varInfo &argType = argumentTypes[arg];

        for(int i = 0; i < candidateCount; ++i){
          if(!ambiguous[i])
            continue;

          if(candidates[i]->getArgument(arg).hasSameTypeAs(argType)){
            for(int i2 = 0; i2 < i; ++i){
              if(!ambiguous[i2]){
                --nonAmbiguousCount;
                ambiguous[i2] = true;
              }
            }

            for(int i2 = (i + 1); i2 < candidateCount; ++i2){
              if(!candidates[i2]->getArgument(arg).hasSameTypeAs(argType)){
                if(!ambiguous[i2]){
                  --nonAmbiguousCount;
                  ambiguous[i2] = true;
                }
              }
            }
          }
        }

        // [-] Clean the error message
        OCCA_CHECK(0 < nonAmbiguousCount,
                   "Ambiguous Function");
      }

      // [-] Clean the error message
      OCCA_CHECK(1 < nonAmbiguousCount,
                 "Ambiguous Function");

      for(int i = 0; i < candidateCount; ++i){
        if(!ambiguous[i])
          return candidates[i];
      }

      return NULL;
    }
    //============================================


    //---[ Var Dependency Graph ]-----------------
#if 0
    int depInfo::startInfo(){
      if(myNode->down == NULL)
        return info;


      return (myNode ? myNode->startNode->startInfo() : depType::none);
    }

    int depInfo::endInfo(){
      return (endNode ? endNode->endInfo() : depType::none);
    }
#endif
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
    baseName("") {}

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
