#include "occaParserTypes.hpp"

namespace occa {
  namespace parserNamespace {
    typeDef::typeDef() :
      up(NULL),

      typeName(""),
      varName(""),
      typeInfo(podTypeDef),

      pointerCount(0),

      bitField(-1),

      typedefing(NULL),
      typedefingBase(NULL) {}

    void typeDef::addVar(varInfo *def){
      if(def->name.size())
        memberVars[def->name] = def;

      allMembers.push_back(def);
      memberInfo.push_back(isVarInfo);
    }

    void typeDef::addType(typeDef *def){
      def->up = this;

      if(def->typeName.size())
        memberTypes[typeName] = def;

      allMembers.push_back(def);
      memberInfo.push_back(isTypeDef);
    }

    typeDef& typeDef::addType(const std::string &newVarName){
      typeDef &def = *(new typeDef);

      def.up      = this;
      def.varName = newVarName;

      memberTypes[newVarName] = &def;
      allMembers.push_back(&def);
      memberInfo.push_back(isTypeDef);

      return def;
    }

    void typeDef::loadFromNode(statement &s,
                               strNode *&n){

      if(n->value == "typedef"){
        typedefing = new typeDef;

        n = n->right;

        const bool isPOD = (!(n->type & structType) ||
                            ((n->right) &&
                             (s.nodeHasDescriptor(n->right))));

        if(isPOD){
          strNode *nRoot = n;

          while(s.nodeHasQualifier(n))
            n = n->right;

          typedefing = s.hasTypeInScope(n->value);

          if(typedefing == NULL){
            std::cout << "Type [" << n->value << "] not defined in scope\n";
            throw 1;
          }

          if(typedefing->typedefing)
            typedefingBase = typedefing->typedefingBase;
          else
            typedefingBase = typedefing;

          while(nRoot != n){
            typeName += nRoot->value + " ";
            nRoot = nRoot->right;
          }

          typeName += n->value;

          varName += n->right->value;

          typedefUsesName = true;

          s.up->scopeTypeMap[varName] = this;
        }
        else{
          typedefing->up = this;
          typedefing->loadFromNode(s, n);

          typedefingBase = typedefing;

          typedefUsesName = false;

          s.up->scopeTypeMap[typedefing->varName] = this;
        }
      }
      else{
        if(n->value == "struct"){
          if(!s.nodeHasSpecifier(n->right))
            typeInfo = structTypeDef;
        }
        else if(n->value == "class")
          typeInfo = classTypeDef;
        else if(n->value == "union")
          typeInfo = unionTypeDef;
        else if(n->value == "enum")
          typeInfo = enumTypeDef;

        if( !(typeInfo & podTypeDef) ){
          // [--]
          // std::cout << "1. HERE\n";
          // n->print();

          if(n->down.size() == 0){
            n = n->right;
            typeName = n->value;

            if(up == NULL)
              s.up->scopeTypeMap[typeName] = this;

            // Empty struct
            if((n->down.size() == 0) &&
               (n->right)            &&
               (n->right->type & endStatement)){
              n->print();

              n = n->right;
              return;
            }
          }

          strNode *nDown    = n->down[0];
          strNode *nDownEnd = lastNode(nDown);

          popAndGoRight(nDown);
          popAndGoLeft(nDownEnd);

          if( !(typeInfo & enumTypeDef) )
            loadStructPartsFromNode(s, nDown);
          else
            loadEnumPartsFromNode(s, nDown);

          if((n->right) &&
             (n->right->type & unknownVariable))
            varName = n->right->value;
        }
        else
          loadStructPartsFromNode(s, n);
      }
    }

    void typeDef::loadStructPartsFromNode(statement &s,
                                          strNode *n){
      bool usingPreviousInfo = false;
      varInfo *lastInfo, *info;

      while(n){
        bool isBitFieldOnly = (n->value == ":");

        const bool isPOD = (usingPreviousInfo ||
                            !(n->type & structType) ||
                            ((n->right) &&
                             (s.nodeHasDescriptor(n->right))));

        if(isPOD){
          info  = new varInfo;
          *info = s.loadVarInfo(n);

          if(usingPreviousInfo){
            info->type        = lastInfo->type;
            info->typeInfo    = lastInfo->typeInfo;
            info->descriptors = lastInfo->descriptors;
          }

          lastInfo = info;

          addVar(info);
        }
        else{
          typeDef &sDef= *(new typeDef);
          sDef.up = this;

          sDef.loadFromNode(s, n);

          addType(&sDef);

          while(n){
            if(n->type & endStatement)
              break;

            n = n->right;
          }
        }

        usingPreviousInfo = (n && (n->value == ","));

        n = (n ? n->right : NULL);
      }
    }

    void typeDef::loadEnumPartsFromNode(statement &s,
                                        strNode *n){
      n->print();
    }

    std::string typeDef::print(const std::string &tab, const int printStyle) const {
      std::string ret = "";

      if(!typedefing && (typeInfo & podTypeDef)){
        if( !(printStyle & typeDefStyle::skipFirstLineIndent) )
          ret += tab;
        else
          ret = "";

        const int heapCount = stackPointerSizes.size();

        const bool hasType = typeName.size();
        const bool hasVar  = varName.size();

        bool needsSpace = false;

        if(hasType){
          ret += typeName;
          needsSpace = true;
        }

        if(pointerCount){
          if(needsSpace){
            ret += " ";
            needsSpace = false;
          }

          for(int i = 0; i < pointerCount; ++i)
            ret += "*";
        }

        if(typeInfo & referenceType){
          if(needsSpace){
            ret += " ";
            needsSpace = false;
          }

          ret += "&";
        }

        if(hasVar){
          if(needsSpace)
            ret += " ";

          needsSpace = true;

          ret += varName;
        }

        if(typeInfo & constPointerType){
          if(needsSpace)
            ret += " ";

          ret += "const";
        }

        for(int i = 0; i < heapCount; ++i){
          ret += "[";
          ret += stackPointerSizes[i];
          ret += "]";
        }

        if(0 <= bitField){
          char sBitField[10];
          sprintf(sBitField, "%d", bitField);

          ret += " : ";
          ret += sBitField;
        }

        if( !(printStyle & typeDefStyle::skipSemicolon) )
          ret += ";";

        return ret;
      }

      if(typedefing){
        if( !(printStyle & typeDefStyle::skipFirstLineIndent) )
          ret += tab;

        ret += "typedef ";

        if(typedefUsesName)
          ret += typeName + " " + varName;
        else
          ret += typedefing->print(tab, (typeDefStyle::skipFirstLineIndent |
                                         typeDefStyle::skipSemicolon));

        ret += ";";

        return ret;
      }

      if( !(printStyle & typeDefStyle::skipFirstLineIndent) )
        ret += tab;

      const int memberCount = allMembers.size();

      if(typeInfo & structTypeDef)
        ret += "struct ";
      else if(typeInfo & classTypeDef)
        ret += "class ";
      else if(typeInfo & unionTypeDef)
        ret += "union ";
      else if(typeInfo & enumTypeDef)
        ret += "enum ";

      if(typeName.size()){
        ret += typeName;

        if(memberCount)
          ret += " ";
      }

      if(memberCount)
        ret += "{\n";

      for(int i = 0; i < memberCount; ++i){
        if(memberInfo[i] & isTypeDef)
          ret += ((typeDef*) allMembers[i])->print(tab + "  ");
        else{
          ret += tab + "  ";
          ret += (std::string) *((varInfo*) allMembers[i]);
          ret += ";";
        }
        ret +=  + "\n";
      }

      if(memberCount){
        ret += tab;
        ret += "}";
      }

      if(varName.size()){
        ret += " ";
        ret += varName;
      }

      if(0 <= bitField){
        char sBitField[10];
        sprintf(sBitField, "%d", bitField);

        ret += " : ";
        ret += sBitField;
      }

      if( !(printStyle & typeDefStyle::skipSemicolon) )
        ret += ";";

      return ret;
    }

    typeDef::operator std::string() const {
      return print();
    }

    std::ostream& operator << (std::ostream &out, const typeDef &def){
      out << (std::string) def;

      return out;
    }
    //==============================================


    //---[ Variable Info ]--------------------------
    varInfo::varInfo() :
      type(NULL),
      name(""),
      typeInfo(0),

      bitField(-1),

      pointerCount(0) {};

    varInfo::varInfo(const varInfo &vi){
      type     = vi.type;
      altType  = vi.altType;
      name     = vi.name;
      typeInfo = vi.typeInfo;

      bitField = vi.bitField;

      pointerCount = vi.pointerCount;

      descriptors       = vi.descriptors;
      stackPointerSizes = vi.stackPointerSizes;

      vars = vi.vars;

      extraInfo = vi.extraInfo;
    }

    std::string varInfo::decoratedType() const {
      const int descriptorCount = descriptors.size();

      if(descriptorCount == 0)
        return (type ? type->typeName : "");

      std::string ret = descriptors[0];

      for(int i = 1; i < descriptorCount; ++i){
        ret += " ";
        ret += descriptors[i];
      }

      ret += " ";
      ret += type->typeName;

      return ret;
    }

    bool varInfo::hasDescriptor(const std::string descriptor) const {
      const int descriptorCount = descriptors.size();

      for(int i = 0; i < descriptorCount; ++i){
        if(descriptors[i] == descriptor)
          return true;
      }

      return false;
    }

    strNode* varInfo::makeStrNodeChain(const int depth,
                                       const int sideDepth) const {
      if(typeInfo & functionPointerType)
        return makeStrNodeChainFromFP(depth, sideDepth);
      else if(typeInfo & functionTypeMask)
        return makeStrNodeChainFromF(depth, sideDepth);

      strNode *nodeRoot = new strNode();
      strNode *nodePos = nodeRoot;

      nodeRoot->depth     = depth;
      nodeRoot->sideDepth = sideDepth;

      const int descriptorCount = descriptors.size();

      for(int i = 0; i < descriptorCount; ++i){
        nodePos       = nodePos->push(descriptors[i]);
        nodePos->type = qualifierType;
      }

      if(type){
        nodePos       = nodePos->push(type->typeName);
        nodePos->type = specifierType;
      }

      if(typeInfo & pointerType){
        if(typeInfo & heapPointerType){
          for(int i = 0; i < pointerCount; ++i){
            nodePos       = nodePos->push("*");
            nodePos->type = keywordType["*"];
          }
        }

        if(typeInfo & constPointerType){
          nodePos       = nodePos->push("const");
          nodePos->type = keywordType["const"];
        }
      }

      if(typeInfo & referenceType){
        nodePos       = nodePos->push("&");
        nodePos->type = keywordType["&"];
      }

      nodePos       = nodePos->push(name);
      nodePos->type = unknownVariable;

      if(typeInfo & stackPointerType){
        const int heapCount = stackPointerSizes.size();

        for(int i = 0; i < heapCount; ++i){
          strNode *downNode = nodePos->pushDown("[");
          downNode->type    = keywordType["["];

          downNode       = downNode->push(stackPointerSizes[i]);
          downNode->type = unknownVariable; // [-] Quick fix

          downNode       = downNode->push("]");
          downNode->type = keywordType["]"];
        }
      }

      if(typeInfo & gotoType){
        nodePos       = nodePos->push(":");
        nodePos->type = keywordType[":"];
      }
      else{
        if(0 <= bitField){
          nodePos       = nodePos->push(":");
          nodePos->type = keywordType[":"];

          char sBitField[10];
          sprintf(sBitField, "%d", bitField);

          nodePos       = nodePos->push(sBitField);
          nodePos->type = presetValue;
        }

        nodePos       = nodePos->push(";");
        nodePos->type = keywordType[";"];
      }

      popAndGoRight(nodeRoot);

      return nodeRoot;
    }

    strNode* varInfo::makeStrNodeChainFromF(const int depth,
                                            const int sideDepth) const {
      strNode *nodeRoot = vars[0]->makeStrNodeChain(depth, sideDepth);

      strNode *nodePos = lastNode(nodeRoot);
      popAndGoLeft(nodePos); // Get rid of the [;]

      nodePos = nodePos->pushDown("(");

      nodePos->type  = keywordType["("];
      nodePos->depth = depth + 1;

      const int varCount = vars.size();

      for(int i = 1; i < varCount; ++i){
        nodePos->right = vars[i]->makeStrNodeChain(depth + 1);
        nodePos = lastNode(nodePos);

        if(i != (varCount - 1)){
          nodePos       = nodePos->push(",");
          nodePos->type = keywordType[","];
        }
      }

      nodePos       = nodePos->push(")");
      nodePos->type = keywordType[")"];

      if(typeInfo & protoType){
        nodePos       = nodePos->push(";");
        nodePos->type = keywordType[";"];
      }

      return nodeRoot;
    }

    strNode* varInfo::makeStrNodeChainFromFP(const int depth,
                                             const int sideDepth) const {
      strNode *nodeRoot = vars[0]->makeStrNodeChain(depth, sideDepth);

      strNode *nodePos = lastNode(nodeRoot);
      popAndGoLeft(nodePos); // Get rid of the [;]

      strNode *nameNode = nodePos;

      nodePos = nodePos->left;
      nodePos->right = NULL;

      //---[ void <(*fp)>(void, void); ]--------
      strNode *downNode = nodePos->pushDown("(");
      downNode->type    = keywordType["("];
      downNode->depth   = depth + 1;

      for(int i = 0; i < pointerCount; ++i){
        downNode       = downNode->push("*");
        downNode->type = keywordType["*"];
      }

      downNode->right = nameNode;
      nameNode->left  = downNode;
      nameNode->right = NULL;

      const int heapCount = stackPointerSizes.size();

      for(int i = 0; i < heapCount; ++i){
        strNode *downNode = nodePos->pushDown("[");
        downNode->type    = keywordType["["];

        downNode       = downNode->push(stackPointerSizes[i]);
        downNode->type = unknownVariable; // [-] Quick fix

        downNode       = downNode->push("]");
        downNode->type = keywordType["]"];
      }

      downNode       = nameNode->push(")");
      downNode->type = keywordType[")"];

      //---[ void (*fp)<(void, void)>; ]--------
      downNode        = nodePos->pushDown("(");
      downNode->type  = keywordType["("];
      downNode->depth = depth + 1;

      const int varCount = vars.size();

      for(int i = 1; i < varCount; ++i){
        downNode->right = vars[i]->makeStrNodeChain(depth + 1);
        downNode = lastNode(downNode);

        if(i != (varCount - 1)){
          downNode       = downNode->push(",");
          downNode->type = keywordType[","];
        }
      }

      downNode       = downNode->push(")");
      downNode->type = keywordType[")"];

      downNode       = downNode->push(";");
      downNode->type = keywordType[";"];

      return nodeRoot;
    }

    varInfo::operator std::string() const {
      if( !(typeInfo & functionTypeMask) )
        return podString();
      else if(typeInfo & functionPointerType)
        return functionPointerString();
      else
        return functionString();
    }

    std::string varInfo::podString() const {
      const int descriptorCount = descriptors.size();
      std::string ret;

      for(int i = 0; i < descriptorCount; ++i){
        ret += descriptors[i];
        ret += ' ';
      }

      if(type){
        ret += type->typeName;
        ret +=  ' ';
      }

      if(typeInfo & pointerType){
        if(typeInfo & heapPointerType){
          for(int i = 0; i < pointerCount; ++i)
            ret += '*';
        }

        if(typeInfo & constPointerType)
          ret += " const ";
      }

      if(typeInfo & referenceType)
        ret += '&';

      ret += name;

      const int heapCount = stackPointerSizes.size();

      for(int i = 0; i < heapCount; ++i){
        ret += '[';
        ret += stackPointerSizes[i];
        ret += ']';
      }

      if(typeInfo & gotoType)
        ret += ':';

      if(0 <= bitField){
        ret += " : ";

        char sBitField[10];
        sprintf(sBitField, "%d", bitField);

        ret += sBitField;
      }

      return ret;
    }

    std::string varInfo::functionString() const {
      std::string ret = decoratedType();

      ret += " ";
      ret += name;
      ret += "(";

      const int varCount = vars.size();

      for(int i = 0; i < varCount; ++i){
        ret += *(vars[i]);

        if(i != (varCount - 1))
          ret += ", ";
      }


      ret += ")";

      return ret;
    }

    std::string varInfo::functionPointerString() const {
      std::string ret = decoratedType();

      //---[ void <(*fp)>(void, void); ]--------
      ret += " (";

      for(int i = 0; i < pointerCount; ++i)
        ret += "*";

      ret += name;

      const int heapCount = stackPointerSizes.size();

      for(int i = 0; i < heapCount; ++i){
        ret += '[';
        ret += stackPointerSizes[i];
        ret += ']';
      }

      ret += ")";

      //---[ void (*fp)<(void, void)>; ]--------
      ret += "(";

      const int varCount = vars.size();

      for(int i = 0; i < varCount; ++i){
        ret += (std::string) *(vars[i]);

        if(i != (varCount - 1))
          ret += ", ";
      }

      ret += ")";

      return ret;
    }

    std::ostream& operator << (std::ostream &out, const varInfo &info){
      out << (std::string) info;
      return out;
    }
    //==============================================
  };
};
