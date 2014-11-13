#include "occaParserStatement.hpp"
#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    //---[ Exp Node ]-------------------------------
    expNode::expNode() :
      sInfo(NULL),

      value(""),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leafInfo(leafType::exp),
      leaves(NULL) {}

    expNode::expNode(statement &s) :
      sInfo(&s),

      value(""),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leafInfo(leafType::exp),
      leaves(NULL) {}

    expNode::expNode(expNode &up_) :
      sInfo(up_.sInfo),

      value(""),
      info(expType::root),

      up(&up_),

      leafCount(0),
      leafInfo(leafType::exp),
      leaves(NULL) {}

    int expNode::getStatementType(){
      if(info & expType::macro_)
        return macroStatementType;

      else if(info & expType::occaFor)
        return keywordType["occaOuterFor0"];

      else if(info & (expType::qualifier |
                      expType::type)){

        if(typeInfo::statementIsATypeInfo(*this, 0))
          return structStatementType;

        varInfo var;
        var.loadFrom(*this, 0);

        if(var.info & varType::var)
          return declareStatementType;
        else if(var.info & varType::functionDec)
          return functionPrototypeType;
        else
          return functionDefinitionType;
      }

      else if((info & (expType::unknown |
                       expType::variable)) &&
              (1 < leafCount) &&
              (leaves[1]->value == ":")){

        return gotoStatementType;
      }

      else if((info == expType::C) &&
              (leaves[0]->value == "{")){

        return blockStatementType;
      }

      return updateStatementType;
    }

    void expNode::loadFromNode(strNode *&nodePos){
      if(nodePos == NULL){
        sInfo->type = invalidStatementType;
        return;
      }

      strNode *nodeRoot = nodePos;

      sInfo->labelStatement(nodePos, this);

      // Don't need to load stuff
      if(sInfo->type & (macroStatementType  |
                        gotoStatementType   |
                        blockStatementType)            ||
         (sInfo->type == keywordType["occaOuterFor0"]) ||
         (sInfo->type == elseStatementType)            ||
         (sInfo->type == doWhileStatementType)){

        return;
      }

      //---[ Special Type ]---
      if(nodeRoot->type & specialKeywordType){
        if((nodeRoot->value == "break")    ||
           (nodeRoot->value == "continue") ||
           (nodeRoot->value == "default")){

          value = nodeRoot->value;
          info  = expType::printValue;
          return;
        }

        // [-] Doesn't support GCC's twisted [Labels as Values]
        if(nodeRoot->value == "goto"){
          value = nodeRoot->right->value;
          info  = expType::goto_;
          return;
        }

        // Case where nodeRoot = [case, return]

        if(nodeRoot->value == "case")
          info = expType::case_;
        else if(nodeRoot->value == "return")
          info = expType::return_;
      }
      //======================

      strNode *newNodeRoot = nodeRoot->cloneTo(nodePos);
      strNode *lastNewNode = lastNode(newNodeRoot);

      if(lastNewNode == NULL)
        newNodeRoot->print();

      splitAndOrganizeNode(newNodeRoot);

      // std::cout << "[" << getBits(sInfo->type) << "] this = " << *this << '\n';

      // Only the root needs to free
      if(up == NULL)
        occa::parserNamespace::free(newNodeRoot);
    }

    void expNode::splitAndOrganizeNode(strNode *nodeRoot){
      initLoadFromNode(nodeRoot);
      initOrganization();

      if(sInfo->type & declareStatementType)
        splitDeclareStatement();

      else if((sInfo->type & (ifStatementType  |
                              forStatementType |
                              whileStatementType)) &&
              (sInfo->type != elseStatementType)){

        splitFlowStatement();
      }

      else if(sInfo->type & functionStatementType)
        splitFunctionStatement();

      else if(sInfo->type & structStatementType)
        splitStructStatement();

      else{
        // Add used vars to varUsedMap
        labelUsedVariables();

        organize();
      }
    }

    void expNode::organize(){
      if(leafCount == 0)
        return;

      organizeLeaves();
    }

    void expNode::splitDeclareStatement(const int flags){
      info = expType::declaration;

      int varCount = 1 + typeInfo::delimeterCount(*this, ",");
      int leafPos  = 0;

      varInfo *firstVar = NULL;

      // Store variables and stuff
      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(expType::root, 0, varCount);

      for(int i = 0; i < varCount; ++i){
        expNode &leaf = newExp[i];
        varInfo &var  = leaf.addVarInfoNode(0);

        int nextLeafPos = var.loadFrom(*this, leafPos, firstVar);

        if(flags & expFlag::addVarToScope){
          if(flags & expFlag::addToParent){
            if(sInfo->up != NULL)
              sInfo->up->addVariable(&var, sInfo);
          }
          else
            sInfo->addVariable(&var);
        }

        if(i == 0){
          leaf.leaves[0]->info |= expType::type;
          firstVar = &var;
        }

        removeNodes(leafPos, nextLeafPos - leafPos);

        int sExpStart = leafPos;
        int sExpEnd   = typeInfo::nextDelimeter(*this, leafPos, ",");

        leafPos = sExpEnd;

        // Don't put the [;]
        if((sExpEnd == leafCount) &&
           (leaves[sExpEnd - 1]->value == ";"))
          --sExpEnd;

        if(sExpStart < sExpEnd){
          leaf.addNodes(0, 1, sExpEnd - sExpStart);

          for(int j = sExpStart; j < sExpEnd; ++j)
            expNode::swap(*leaf.leaves[j - sExpStart + 1], *leaves[j]);

          leaf.labelUsedVariables();
          leaf.organizeLeaves();
        }

        if(leafPos < leafCount)
          removeNode(leafPos);
      }

      expNode::swap(*this, newExp);
    }

    void expNode::splitFlowStatement(){
      info = expType::checkSInfo;

      expNode &expDown = *(leaves[1]);

      int statementCount = 1 + typeInfo::delimeterCount(expDown, ";");

      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(expType::root, 0, statementCount);

      int leafPos = 0;

      for(int i = 0; i < statementCount; ++i){
        expNode &leaf = newExp[i];

        int nextLeafPos = typeInfo::nextDelimeter(expDown, leafPos, ";");

        if(leafPos < nextLeafPos){
          leaf.addNodes(0, 0, (nextLeafPos - leafPos));

          for(int j = 0; j < leaf.leafCount; ++j){
            delete leaf.leaves[j];

            leaf.leaves[j]     = expDown.leaves[leafPos + j];
            leaf.leaves[j]->up = &leaf;
          }

          if(!(sInfo->type & forStatementType) || (i != 0))
            leaf.organize();
          else
            leaf.splitDeclareStatement(expFlag::addVarToScope);
        }

        leafPos = (nextLeafPos + 1);
      }

      expNode::swap(*this, newExp);
    }

    void expNode::splitFunctionStatement(const int flags){
      if(sInfo->type & functionDefinitionType)
        info = (expType::function | expType::declaration);
      else
        info = (expType::function | expType::prototype);

      if(leafCount == 0)
        return;

      varInfo &var = addVarInfoNode(0);
      int leafPos  = var.loadFrom(*this, 1);

      if((flags & expFlag::addVarToScope) &&
         (sInfo->up != NULL)              &&
         (sInfo->up->scopeVarMap.find(var.name) ==
          sInfo->up->scopeVarMap.end())){

        sInfo->up->addVariable(&var);
      }

      removeNodes(1, leafPos);
    }

    void expNode::splitStructStatement(const int flags){
      info = expType::struct_;

      // Store type
      expNode newExp(*sInfo);
      newExp.info = info;

      typeInfo &type = newExp.addTypeInfoNode(0);

      int leafPos = type.loadFrom(*this, 0);

      if(flags & expFlag::addTypeToScope){
        if(flags & expFlag::addToParent){
          if(sInfo->up != NULL)
            sInfo->up->addType(type);
        }
        else
          sInfo->addType(type);
      }

      expNode::swap(*this, newExp);
    }

    void expNode::initLoadFromNode(strNode *nodeRoot){
      strNode *nodePos = nodeRoot;

      while(nodePos){
        ++leafCount;
        nodePos = nodePos->right;
      }

      if(leafCount == 0)
        return;

      nodePos = nodeRoot;

      leaves = new expNode*[leafCount];
      int leafPos = 0;

      while(nodePos){
        expNode *&leaf = leaves[leafPos++];

        leaf        = new expNode(*this);
        leaf->value = nodePos->value;

        if(nodePos->type & unknownVariable){
          varInfo *nodeVar = sInfo->hasVariableInScope(nodePos->value);

          if(nodeVar){
            if( !(nodeVar->info & varType::functionType) )
              leaf->info = expType::variable;
            else
              leaf->info = expType::function;
          }
          else{
            typeInfo *nodeType = sInfo->hasTypeInScope(nodePos->value);

            if(!nodeType)
              leaf->info = expType::unknown;
            else
              leaf->info = expType::type;
          }
        }

        else if(nodePos->type & presetValue){
          leaf->info = expType::presetValue;
        }

        else if(nodePos->type & descriptorType){
          if(nodePos->type == keywordType["long"]){
            if((nodePos->right) &&
               (sInfo->hasTypeInScope(nodePos->right->value))){

              leaf->info = expType::qualifier;
            }
            else
              leaf->info = expType::type;
          }
          else if(nodePos->type & (qualifierType | structType))
            leaf->info = expType::qualifier;
          else
            leaf->info = expType::type;

          // For [*] and [&]
          if(nodePos->type & operatorType)
            leaf->info |= expType::operator_;
        }

        else if(nodePos->type & structType){
          leaf->info = expType::qualifier;
        }

        else if(nodePos->type & operatorType){
          leaf->info = expType::operator_;
        }

        else if(nodePos->type & startSection){
          leaf->info  = expType::C;

          if(nodePos->down)
            leaf->initLoadFromNode(nodePos->down);
        }

        else
          leaf->info = expType::printValue;

        if(nodePos->type == 0){
          delete leaf;
          --leafPos;
        }

        nodePos = nodePos->right;
      }
    }

    void expNode::initOrganization(){
      if(leafCount == 0)
        return;

      // Init ()'s bottom -> up
      // Organize leaves bottom -> up
      for(int i = 0; i < leafCount; ++i){
        if((leaves[i]->leafCount) &&
           !(leaves[i]->info & (expType::varInfo |
                                expType::typeInfo))){

          leaves[i]->initOrganization();
        }
      }

      //---[ Level 0 ]------
      // [a][::][b]
      mergeNamespaces();

      // [(class)]
      labelCasts();

      // const int [*] x
      labelReferenceQualifiers();
      //====================
    }

    void expNode::organizeLeaves(){
      if(info & (expType::varInfo |
                 expType::typeInfo))
        return;

      // Organize leaves bottom -> up
      for(int i = 0; i < leafCount; ++i){
        if((leaves[i]->leafCount) &&
           !(leaves[i]->info & (expType::varInfo |
                                expType::typeInfo))){

          leaves[i]->organizeLeaves();
        }
      }

      //---[ Level 1 ]------
      // class(...), class{1,2,3}
      mergeClassConstructs();

      // static_cast<>()
      mergeCasts();

      // [max(a,b)]
      mergeFunctionCalls();

      // a[3]
      mergeArrays();

      organizeLeaves(1);
      //====================

      //---[ Level 2 ]------
      // (class) x
      mergeClassCasts();

      // sizeof x
      mergeSizeOf();

      // new, new [], delete, delete []
      mergeNewsAndDeletes();

      organizeLeaves(2);
      //====================

      //---[ Level 3-14 ]---
      for(int i = 3; i <= 14; ++i)
        organizeLeaves(i);
      //====================

      //---[ Level 15 ]-----
      // throw x
      mergeThrows();
      //====================

      //---[ Level 16 ]-----
      organizeLeaves(16);
      //====================
    }

    void expNode::organizeLeaves(const int level){
      int leafPos = 0;

      while(leafPos < leafCount){
        if((leaves[leafPos]->leafCount) ||
           (leaves[leafPos]->info == expType::qualifier)){

          ++leafPos;
          continue;
        }

        opLevelMapIterator it = opLevelMap[level].find(leaves[leafPos]->value);

        if(it == opLevelMap[level].end()){
          ++leafPos;
          continue;
        }

        const int levelType = it->second;

        if(levelType & unitaryOperatorType){
          bool updateNow = true;

          // Cases:  1 + [-]1
          //         (+1)
          if((leaves[leafPos]->value.size() == 1) &&
             ((leaves[leafPos]->value[0] == '+') ||
              (leaves[leafPos]->value[0] == '-'))){

            if(leafPos &&
               ((leaves[leafPos - 1]->leafCount != 0) ||
                !(leaves[leafPos - 1]->info & expType::operator_))){

              updateNow = false;
            }
          }

          if(updateNow){
            int target = leafPos + ((levelType & lUnitaryOperatorType) ?
                                    1 : -1);

            if((target < 0) || (leafCount <= target)){
              ++leafPos;
            }
            else{
              if(levelType & lUnitaryOperatorType)
                leafPos = mergeLeftUnary(leafPos);
              else
                leafPos = mergeRightUnary(leafPos);
            }
          }
          else
            ++leafPos;
        }
        else if(levelType & binaryOperatorType)
          leafPos = mergeBinary(leafPos);
        else if(levelType & ternaryOperatorType)
          leafPos = mergeTernary(leafPos);
        else
          ++leafPos;
      }
    }

    int expNode::mergeRange(const int newLeafType,
                            const int leafPosStart,
                            const int leafPosEnd){
      expNode *newLeaf = new expNode(*this);

      newLeaf->up        = this;
      newLeaf->info      = newLeafType;
      newLeaf->leafCount = (leafPosEnd - leafPosStart + 1);
      newLeaf->leaves    = new expNode*[newLeaf->leafCount];

      for(int i = 0; i < newLeaf->leafCount; ++i){
        newLeaf->leaves[i]     = leaves[leafPosStart + i];
        newLeaf->leaves[i]->up = newLeaf;
      }

      leaves[leafPosStart] = newLeaf;

      for(int i = (leafPosEnd + 1); i < leafCount; ++i)
        leaves[leafPosStart + i - leafPosEnd] = leaves[i];

      leafCount -= (newLeaf->leafCount - 1);

      return (leafPosStart + 1);
    }

    // [a][::][b]
    void expNode::mergeNamespaces(){
      int leafPos = 0;

      while(leafPos < leafCount){
        if(leaves[leafPos]->value == "::"){
          // [-] Change after keeping track of namespaces
          int leafPosStart = leafPos - (leafPos &&
                                        (leaves[leafPos - 1]->info & expType::unknown));
          // int leafPosStart = leafPos - (leafPos &&
          //                               (leaves[leafPos - 1]->info & expType::namespace_));
          int leafPosEnd   = leafPos + 2;

          while((leafPosEnd < leafCount) &&
                (leaves[leafPosEnd]->value == "::"))
            leafPosEnd += 2;

          --leafPosEnd;

          leafPos = mergeRange(expType::type | expType::namespace_,
                               leafPosStart, leafPosEnd);
        }
        else
          ++leafPos;
      }
    }

    // const int [*] x
    void expNode::labelReferenceQualifiers(){
      int leafPos = 0;

      while(leafPos < leafCount){
        if(leaves[leafPos]->info == (expType::operator_ |
                                     expType::qualifier)){
          if(leafPos == 0){
            leaves[leafPos]->info = expType::qualifier;
          }
          else{
            expNode &lLeaf = *(leaves[leafPos - 1]);

            if(lLeaf.info & expType::qualifier){
              leaves[leafPos]->info = expType::qualifier;
            }

            else if(lLeaf.info & expType::unknown){
              if(!sInfo->hasTypeInScope(lLeaf.value))
                leaves[leafPos]->info = expType::operator_;
              else
                leaves[leafPos]->info = expType::qualifier;
            }

            else if(lLeaf.info & (expType::L           |
                                  expType::R           |
                                  expType::presetValue |
                                  expType::variable    |
                                  expType::function)){

              leaves[leafPos]->info = expType::operator_;
            }

            else if((lLeaf.info & expType::C)      &&
                    !(lLeaf.info & expType::cast_) &&
                    (lLeaf.value != "{")){

              leaves[leafPos]->info = expType::operator_;
            }

            else{
              leaves[leafPos]->info = expType::qualifier;
            }
          }
        }

        ++leafPos;
      }
    }

    // [(class)]
    void expNode::labelCasts(){
      // Don't mistake:
      //   int main(int) -> int main[(int)]
      if(sInfo->type & functionStatementType)
        return;

      int leafPos = 0;

      while(leafPos < leafCount){
        if((leaves[leafPos]->value == "(")       &&
           (leaves[leafPos]->leafCount)          &&
           (leaves[leafPos]->leaves[0]->info & (expType::type      |
                                                expType::qualifier |
                                                expType::typeInfo))){

          leaves[leafPos]->info |= expType::cast_;
        }

        ++leafPos;
      }
    }

    // Add used vars to varUsedMap
    void expNode::labelUsedVariables(){
      if(info & (expType::varInfo |
                 expType::typeInfo))
        return;

      // Organize leaves bottom -> up
      for(int i = 0; i < leafCount; ++i){
        if((leaves[i]->leafCount) &&
           !(leaves[i]->info & (expType::varInfo |
                                expType::typeInfo))){

          leaves[i]->labelUsedVariables();
        }
      }

      expNode &flatRoot = *(makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        if(flatRoot[i].info & (expType::variable |
                               expType::function)){
          varInfo &var = *(sInfo->hasVariableInScope(flatRoot[i].value));

          sInfo->varUsedMap[&var].push(sInfo);
        }
      }

      freeFlatHandle(flatRoot);
    }

    // class(...), class{1,2,3}
    void expNode::mergeClassConstructs(){
    }

    // static_cast<>()
    void expNode::mergeCasts(){
    }

    // [max(a,b)]
    void expNode::mergeFunctionCalls(){
      int leafPos = 0;

      while(leafPos < leafCount){
        if((leaves[leafPos]->info  == expType::C) &&
           (leaves[leafPos]->value == "(")){

          if((leafPos) &&
             (leaves[leafPos - 1]->info & expType::function)){
            expNode &fNode    = *(leaves[leafPos - 1]);
            expNode &argsNode = *(leaves[leafPos    ]);

            fNode.addNode(argsNode.info);

            delete fNode.leaves[0];
            fNode.leaves[0] = &argsNode;

            removeNode(leafPos);
            --leafPos;
          }
        }

        ++leafPos;
      }
    }

    void expNode::mergeArguments(){
      for(int i = 0; i < leafCount; i += 2){
        leaves[i/2] = leaves[i];

        if((i + 1) < leafCount)
          freeLeaf(i + 1);
      }

      leafCount = ((leafCount / 2) + 1);
    }

    // a[2]
    void expNode::mergeArrays(){
      int leafPos = 0;

      while(leafPos < leafCount){
        if((leaves[leafPos]->info & expType::C) &&
           (leaves[leafPos]->value == "[")){

          int brackets = 0;

          while(((leafPos + brackets) < leafCount) &&
                (leaves[leafPos + brackets]->info & expType::C) &&
                (leaves[leafPos + brackets]->value == "[")){

            ++brackets;
          }

          const bool inserting = ((leaves[leafPos - 1]->info & expType::variable) &&
                                  leaves[leafPos - 1]->leafCount);

          expNode *newLeaf;

          if(inserting){
            newLeaf = leaves[leafPos - 1];
          }
          else{
            newLeaf = new expNode(*this);

            newLeaf->up        = this;
            newLeaf->info      = expType::variable;
            newLeaf->leafCount = 2;
            newLeaf->leaves    = new expNode*[2];
          }

          expNode *sNewLeaf = new expNode(*newLeaf);

          sNewLeaf->up        = newLeaf;
          sNewLeaf->info      = expType::qualifier;
          sNewLeaf->leafCount = brackets;
          sNewLeaf->leaves    = new expNode*[brackets];

          if(inserting){
            newLeaf->addNode(expType::qualifier, newLeaf->leafCount);
            newLeaf->leaves[newLeaf->leafCount - 1] = sNewLeaf;
          }
          else{
            newLeaf->leaves[0] = leaves[leafPos - 1];
            newLeaf->leaves[1] = sNewLeaf;

            leaves[leafPos - 1] = newLeaf;
          }

          for(int i = 0; i < brackets; ++i)
            sNewLeaf->leaves[i] = leaves[leafPos + i];

          for(int i = (leafPos + brackets); i < leafCount; ++i)
            leaves[i - brackets] = leaves[i];

          ++leafPos;
          leafCount -= brackets;
        }
        else
          ++leafPos;
      }
    }

    // (class) x
    void expNode::mergeClassCasts(){
    }

    // sizeof x
    void expNode::mergeSizeOf(){
    }

    // new, new [], delete, delete []
    void expNode::mergeNewsAndDeletes(){
    }

    // throw x
    void expNode::mergeThrows(){
    }

    // [++]i
    int expNode::mergeLeftUnary(const int leafPos){
      expNode *leaf  = leaves[leafPos];
      expNode *sLeaf = leaves[leafPos + 1];

      for(int i = (leafPos + 2); i < leafCount; ++i)
        leaves[i - 1] = leaves[i];

      --leafCount;

      leaf->info      = expType::L;
      leaf->leafCount = 1;
      leaf->leaves    = new expNode*[1];
      leaf->leaves[0] = sLeaf;

      sLeaf->up = leaf;

      return (leafPos + 1);
    }

    // i[++]
    int expNode::mergeRightUnary(const int leafPos){
      expNode *leaf  = leaves[leafPos];
      expNode *sLeaf = leaves[leafPos - 1];

      leaves[leafPos - 1] = leaf;

      for(int i = (leafPos + 1); i < leafCount; ++i)
        leaves[i - 1] = leaves[i];

      --leafCount;

      leaf->info      = expType::R;
      leaf->leafCount = 1;
      leaf->leaves    = new expNode*[1];
      leaf->leaves[0] = sLeaf;

      sLeaf->up = leaf;

      return (leafPos + 1);
    }

    // a [+] b
    int expNode::mergeBinary(const int leafPos){
      expNode *leaf   = leaves[leafPos];
      expNode *sLeafL = leaves[leafPos - 1];
      expNode *sLeafR = leaves[leafPos + 1];

      leaves[leafPos - 1] = leaf;

      for(int i = (leafPos + 2); i < leafCount; ++i)
        leaves[i - 2] = leaves[i];

      leafCount -= 2;

      leaf->info      = (expType::L | expType::R);
      leaf->leafCount = 2;
      leaf->leaves    = new expNode*[2];
      leaf->leaves[0] = sLeafL;
      leaf->leaves[1] = sLeafR;

      sLeafL->up = leaf;
      sLeafR->up = leaf;

      return leafPos;
    }

    // a [?] b : c
    int expNode::mergeTernary(const int leafPos){
      expNode *leaf   = leaves[leafPos];
      expNode *sLeafL = leaves[leafPos - 1];
      expNode *sLeafC = leaves[leafPos + 1];
      expNode *sLeafR = leaves[leafPos + 3];

      leaves[leafPos - 1] = leaf;

      leafCount -= 4;

      for(int i = leafPos; i < leafCount; ++i)
        leaves[i] = leaves[i + 4];

      leaf->info      = (expType::L | expType::C | expType::R);
      leaf->leafCount = 3;
      leaf->leaves    = new expNode*[3];
      leaf->leaves[0] = sLeafL;
      leaf->leaves[1] = sLeafC;
      leaf->leaves[2] = sLeafR;

      sLeafL->up = leaf;
      sLeafC->up = leaf;
      sLeafR->up = leaf;

      return leafPos;
    }

    //---[ Custom Type Info ]---------
    bool expNode::qualifierEndsWithStar() const {
      if( !(info & expType::qualifier) )
        return false;

      if(leafCount)
        return leaves[leafCount - 1]->qualifierEndsWithStar();
      else
        return (value == "*");
    }

    bool expNode::typeEndsWithStar() const {
      if( !(info & expType::type) ||
          (leafCount == 0) )
        return false;

      if(leaves[leafCount - 1]->info & expType::qualifier)
        return leaves[leafCount - 1]->qualifierEndsWithStar();

      return false;
    }

    bool expNode::hasAnArrayQualifier(const int pos) const {
      if( !(info & expType::qualifier) ||
          (leafCount <= pos) )
        return false;

      return ((leaves[pos]->value == "*") ||
              (leaves[pos]->value == "&") ||
              (leaves[pos]->value == "["));
    }
    //================================

    void expNode::swap(expNode &a, expNode &b){
      swapValues(a.sInfo, b.sInfo);

      swapValues(a.value, b.value);
      swapValues(a.info , b.info);

      swapValues(a.up, b.up);

      swapValues(a.leafCount, b.leafCount);
      swapValues(a.leaves   , b.leaves);

      for(int i = 0; i < a.leafCount; ++i)
        a.leaves[i]->up = &a;

      for(int i = 0; i < b.leafCount; ++i)
        b.leaves[i]->up = &b;
    }

    expNode* expNode::clone(statement &s){
      expNode &newRoot = *(new expNode(s));

      newRoot.info = info;

      if(info & (expType::varInfo | expType::typeInfo)){
        if(info & expType::varInfo){
          varInfo &var = newRoot.addVarInfoNode();
          var = getVarInfo().clone();
        }
        if(info & expType::typeInfo){
          typeInfo &type = newRoot.addTypeInfoNode();
          type = getTypeInfo().clone();
        }
      }
      else {
        newRoot.value     = value;
        newRoot.leafCount = leafCount;

        if(leafCount){
          newRoot.leaves = new expNode*[leafCount];

          for(int i = 0; i < leafCount; ++i){
            newRoot.leaves[i]     = clone(leaves[i]);
            newRoot.leaves[i]->up = &newRoot;
          }
        }
      }

      return &newRoot;
    }

    expNode* expNode::clone(expNode *original){
      expNode &newLeaf = *(new expNode(*this));
      expNode &o = *original;

      newLeaf.info = o.info;

      if(o.info & (expType::varInfo | expType::typeInfo)){
        if(o.info & expType::varInfo){
          varInfo &var = newLeaf.addVarInfoNode();
          var = o.getVarInfo().clone();
        }
        else if(o.info & expType::typeInfo){
          typeInfo &type = newLeaf.addTypeInfoNode();
          type = o.getTypeInfo().clone();
        }
      }
      else {
        newLeaf.value     = o.value;
        newLeaf.leafCount = o.leafCount;

        if(o.leafCount){
          newLeaf.leaves = new expNode*[o.leafCount];

          for(int i = 0; i < o.leafCount; ++i){
            newLeaf.leaves[i]     = o.clone(o.leaves[i]);
            newLeaf.leaves[i]->up = &newLeaf;
          }
        }
      }

      return &newLeaf;
    }

    void expNode::cloneTo(expNode &newRoot){
      newRoot.info  = info;

      if(info & (expType::varInfo | expType::typeInfo)){
        if(info & expType::varInfo){
          varInfo &var = newRoot.addVarInfoNode();
          var = getVarInfo().clone();
        }
        else if(info & expType::typeInfo){
          typeInfo &type = newRoot.addTypeInfoNode();
          type = getTypeInfo().clone();
        }
      }
      else {
        newRoot.value     = value;
        newRoot.leafCount = leafCount;

        if(leafCount){
          newRoot.leaves = new expNode*[leafCount];

          for(int i = 0; i < leafCount; ++i){
            newRoot.leaves[i]     = clone(leaves[i]);
            newRoot.leaves[i]->up = &newRoot;
          }
        }
      }
    }

    expNode* expNode::lastLeaf(){
      if(leafCount != 0)
        return leaves[leafCount - 1];

      return NULL;
    }

    //---[ Exp Info ]-----------------
    int expNode::depth(){
      int depth_ = 0;

      while(up)
        ++depth_;

      return depth_;
    }

    int expNode::whichLeafAmI(){
      if(up){
        const int upLeafCount = up->leafCount;

        for(int i = 0; i < upLeafCount; ++i)
          if(up->leaves[i] == this)
            return i;
      }

      return -1;
    }

    int expNode::nestedLeafCount(){
      if(info & (expType::varInfo |
                 expType::typeInfo) ){

        return 0;
      }

      int ret = leafCount;

      for(int i = 0; i < leafCount; ++i){
        if(leaves[i]->leafCount)
          ret += leaves[i]->nestedLeafCount();
      }

      return ret;
    }

    expNode* expNode::makeFlatHandle(){
      if(leafCount == 0)
        return NULL;

      expNode &flatNode = *(new expNode(*sInfo));

      flatNode.info   = expType::printLeaves;
      flatNode.leaves = new expNode*[nestedLeafCount()];

      int offset = 0;
      makeFlatHandle(offset, flatNode.leaves);

      flatNode.leafCount = offset;

      return &flatNode;
    }

    void expNode::makeFlatHandle(int &offset,
                                 expNode **flatLeaves){
      if(info & (expType::varInfo |
                 expType::typeInfo) ){

        return;
      }

      for(int i = 0; i < leafCount; ++i){
        switch(leaves[i]->info){
        case (expType::L):{
          flatLeaves[offset++] = leaves[i];
          flatLeaves[offset++] = leaves[i]->leaves[0];
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);

          break;
        }

        case (expType::R):{
          flatLeaves[offset++] = leaves[i]->leaves[0];
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i];

          break;
        }

        case (expType::L | expType::R):{
          flatLeaves[offset++] = leaves[i]->leaves[0];
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i];
          flatLeaves[offset++] = leaves[i]->leaves[1];
          leaves[i]->leaves[1]->makeFlatHandle(offset, flatLeaves);

          break;
        }

        case (expType::L | expType::C | expType::R):{
          flatLeaves[offset++] = leaves[i]->leaves[0];
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i]->leaves[1];
          leaves[i]->leaves[1]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i]->leaves[2];
          leaves[i]->leaves[2]->makeFlatHandle(offset, flatLeaves);

          break;
        }
        default:
          flatLeaves[offset++] = leaves[i];

          leaves[i]->makeFlatHandle(offset, flatLeaves);

          break;
        }
      }
    }

    void expNode::freeFlatHandle(expNode &flatRoot){
      delete [] flatRoot.leaves;
      delete &flatRoot;
    }

    void expNode::addNodes(const int info_,
                           const int pos,
                           const int count){

      expNode **newLeaves = new expNode*[leafCount + count];

      //---[ Add Leaves ]-----
      for(int i = 0; i < pos; ++i)
        newLeaves[i] = leaves[i];

      for(int i = pos; i < (pos + count); ++i){
        newLeaves[i] = new expNode(*this);

        newLeaves[i]->info      = info_;
        newLeaves[i]->leafCount = 0;
        newLeaves[i]->leaves    = NULL;
      }

      for(int i = pos; i < leafCount; ++i)
        newLeaves[i + count] = leaves[i];
      //======================

      if(leafCount)
        delete [] leaves;

      leaves = newLeaves;

      leafCount += count;
    }

    void expNode::addNode(const int info_,
                          const int pos){
      addNodes(info_, pos, 1);
    }

    varInfo& expNode::addVarInfoNode(){
      addNode(0);

      varInfo **varLeaves = (varInfo**) leaves;
      varInfo *&varLeaf   = varLeaves[0];

      varLeaf = new varInfo;
      return *varLeaf;
    }

    varInfo& expNode::addVarInfoNode(const int pos){
      addNode(expType::varInfo, pos);
      return leaves[pos]->addVarInfoNode();
    }

    typeInfo& expNode::addTypeInfoNode(){
      addNode(0);

      typeInfo **typeLeaves = (typeInfo**) leaves;
      typeInfo *&typeLeaf   = typeLeaves[0];

      typeLeaf = new typeInfo;
      return *typeLeaf;
    }

    typeInfo& expNode::addTypeInfoNode(const int pos){
      addNode(expType::typeInfo, pos);
      return leaves[pos]->addTypeInfoNode();
    }

    varInfo& expNode::getVarInfo(){
      return *((varInfo*) leaves[0]);
    }

    const varInfo& expNode::cGetVarInfo() const {
      return *((const varInfo*) leaves[0]);
    }

    varInfo& expNode::getVarInfo(const int pos){
      varInfo **varLeaves = (varInfo**) leaves[pos]->leaves;
      varInfo *&varLeaf   = varLeaves[0];

      return *varLeaf;
    }

    const varInfo& expNode::cGetVarInfo(const int pos) const {
      const varInfo **varLeaves = (const varInfo**) leaves[pos]->leaves;
      const varInfo *&varLeaf   = varLeaves[0];

      return *varLeaf;
    }

    typeInfo& expNode::getTypeInfo(){
      return *((typeInfo*) leaves[0]);
    }

    const typeInfo& expNode::cGetTypeInfo() const {
      return *((const typeInfo*) leaves[0]);
    }

    typeInfo& expNode::getTypeInfo(const int pos){
      typeInfo **typeLeaves = (typeInfo**) leaves[pos]->leaves;
      typeInfo *&typeLeaf   = typeLeaves[0];

      return *typeLeaf;
    }

    const typeInfo& expNode::cGetTypeInfo(const int pos) const {
      const typeInfo **typeLeaves = (const typeInfo**) leaves[pos]->leaves;
      const typeInfo *&typeLeaf   = typeLeaves[0];

      return *typeLeaf;
    }

    void expNode::removeNodes(const int pos, const int count){
      const int removed = (((pos + count) <= leafCount) ?
                           count : (leafCount - pos));

      for(int i = (pos + count); i < leafCount; ++i)
        leaves[i - count] = leaves[i];

      leafCount -= removed;
    }

    void expNode::removeNode(const int pos){
      removeNodes(pos, 1);
    }

    void expNode::convertTo(const int info_){
      if(info & expType::declaration){
        if(info_ & expType::variable){
          info = expType::variable;

          leafCount = 2;

          expNode *varNode = leaves[1]->leaves[0]->clone(*sInfo);

          leaves[1]->free();
          leaves[1] = varNode;
        }
      }
    }

    bool expNode::hasQualifier(const std::string &qualifier) const {
      if(info & expType::type){
        if(!leafCount ||
           !(leaves[0]->info & expType::qualifier))
          return false;

        return leaves[0]->hasQualifier(qualifier);
      }
      else if(info & expType::qualifier){
        if(leafCount){
          for(int i = 0; i < leafCount; ++i){
            if(leaves[i]->value == qualifier)
              return true;
          }

          return false;
        }
        else
          return value == qualifier;
      }
      else if(info & expType::variable){
        if((leafCount) &&
           (leaves[0]->info & expType::type)){

          return leaves[0]->hasQualifier(qualifier);
        }
        else
          return false;
      }

      return false;
    }

    void expNode::addQualifier(const std::string &qualifier,
                               const int pos){
      if(info & expType::variable){
        if(leafCount){
          expNode &lqNode = *(leaves[0]);

          if( !(lqNode.info & expType::type) ){
            std::cout << "5. Error on:" << *this << '\n';
            throw 1;
          }

          if( !(lqNode.leaves[0]->info & expType::qualifier) )
            lqNode.addNode(expType::qualifier, 0);

          expNode &qNode = *(lqNode.leaves[0]);

          qNode.addNode(expType::qualifier, pos);
          qNode.leaves[pos]->value = qualifier;
        }
      }
    }

    void expNode::addPostQualifier(const std::string &qualifier,
                                   const int pos){
      if(info & expType::variable){
        if(leafCount){
          expNode &lqNode = *(leaves[0]);

          if( !(lqNode.info & expType::type) ){
            std::cout << "5. Error on:" << *this << '\n';
            throw 1;
          }

          if( !(lqNode.lastLeaf()->info & expType::qualifier) )
            lqNode.addNode(expType::qualifier, lqNode.leafCount);

          expNode &qNode = *(lqNode.lastLeaf());

          qNode.addNode(expType::qualifier, pos);
          qNode.leaves[pos]->value = qualifier;
        }
      }
    }

    void expNode::removeQualifier(const std::string &qualifier){
      if(info & expType::type){
        if(leafCount){
          expNode &qNode = *(leaves[0]);

          if( !(qNode.info & expType::qualifier) )
            return;

          for(int i = 0; i < qNode.leafCount; ++i){
            if(qNode.leaves[i]->value == qualifier){
              qNode.removeNode(i);

              // Erase if there are no qualifiers
              if(qNode.leafCount == 0)
                removeNode(0);

              return;
            }
          }
        }
      }
    }

    void expNode::changeType(const std::string &newType){
      if(info & expType::variable){
        if(leafCount){
          const bool hasLQualifier = (leaves[0]->info & expType::type);

          if(hasLQualifier)
            leaves[0]->changeType(newType);
        }
      }
      else if(info & expType::type){
        if(leaves[0]->info & expType::type)
          leaves[0]->value = newType;
        else
          leaves[1]->value = newType;
      }
      else if(info & expType::declaration){
        if(leafCount &&
           (leaves[0]->info & expType::type)){

          leaves[0]->changeType(newType);
        }
      }
    }

    int expNode::getVariableCount() const {
      if(info & expType::declaration){
        return leafCount;
      }

      return 0;
    }

    bool expNode::variableHasInit(const int pos) const {
      if(info & expType::declaration){
        const expNode &varNode = *(getVariableNode(pos));

        return (varNode.leafCount &&
                (varNode.leaves[0]->value == "="));
      }

      return false;
    }

    expNode* expNode::getVariableNode(const int pos) const {
      if(info & expType::declaration){
        return leaves[pos];
      }

      return NULL;
    }

    expNode* expNode::getVariableInfoNode(const int pos) const {
      if(info & expType::declaration){
        const expNode &varNode = *(getVariableNode(pos));

        if(varNode.leaves[0]->info & expType::varInfo)
          return varNode.leaves[0];
        else if(varNode.leafCount &&
                (varNode.leaves[0]->value == "=")){

          return varNode.leaves[0]->leaves[0];
        }
      }

      return NULL;
    }

    expNode* expNode::getVariableInitNode(const int pos) const {
      if(info & expType::declaration){
        if(variableHasInit(pos)){
          const expNode &varNode = *(getVariableNode(pos));

          if(varNode.leafCount &&
             (varNode.leaves[0]->value == "=")){

            return varNode.leaves[0]->leaves[1];
          }
        }
      }

      return NULL;
    }

    std::string expNode::getVariableName(const int pos) const {
      if(info & expType::declaration){
        expNode &leaf = *(leaves[pos]);

        if(leaf.info & expType::varInfo){
          return leaf.getVarInfo().name;
        }
        else if(leaf.leafCount &&
                (leaf[0].value == "=")){

          return leaf[0].getVarInfo(0).name;
        }
      }
      else if(info & expType::variable){
        if(leafCount){
          const bool hasLQualifier = (leaves[0]->info & expType::type);

          return leaves[hasLQualifier]->value;
        }
        else
          return value;
      }

      return "";
    }
    //================================

    void expNode::freeLeaf(const int leafPos){
      leaves[leafPos]->free();
      delete leaves[leafPos];
    }

    // [-] Not properly done for varInfo and typeInfo
    void expNode::free(){
      if(info & (expType::varInfo | expType::typeInfo)){

        if(info & expType::varInfo)
          delete (varInfo*) leaves[0];
        if(info & expType::typeInfo)
          delete (typeInfo*) leaves[0];

        delete [] leaves;

        return;
      }

      for(int i = 0; i < leafCount; ++i){
        leaves[i]->free();
        delete leaves[i];
      }

      leafCount = 0;
      delete [] leaves;
    }

    void expNode::print(const std::string &tab){
      std::cout << tab << "[" << getBits(info) << "] " << value << '\n';

      if( !(info & (expType::varInfo |
                    expType::typeInfo)) ){

        for(int i = 0; i < leafCount; ++i)
          leaves[i]->print(tab + "    ");
      }
      else if(info & expType::varInfo){
        std::cout << tab << "    [varInfo] " << getVarInfo() << '\n';
      }
      else if(info & expType::typeInfo){
        std::cout << tab << "    [typeInfo]\n" << getTypeInfo().toString(tab + "        ") << '\n';
      }
    }

    void expNode::printOn(std::ostream &out,
                          const std::string &tab,
                          const int flags){
      switch(info){
      case (expType::root):{
        out << tab;

        for(int i = 0; i < leafCount; ++i)
          out << *(leaves[i]);

        break;
      }

      case (expType::L):{
        out << value << *(leaves[0]);

        break;
      }

      case (expType::R):{
        out << *(leaves[0]) << value;

        break;
      }

      case (expType::L | expType::R):{
        if((value != ".") && (value != "->"))
          out << *(leaves[0]) << ' ' << value << ' ' << *(leaves[1]);
        else
          out << *(leaves[0]) << value << *(leaves[1]);

        break;
      }

      case (expType::L | expType::C | expType::R):{
        out << *(leaves[0]) << '?' << *(leaves[1]) << ':' << *(leaves[2]);

        break;
      }

      case (expType::C):{
        const char startChar = value[0];

        out << startChar;

        for(int i = 0; i < leafCount; ++i)
          out << *(leaves[i]);

        out << segmentPair(startChar);

        break;
      }

      case (expType::qualifier):{
        if(leafCount){
          out << *(leaves[0]);

          for(int i = 1; i < leafCount; ++i){
            if( !hasAnArrayQualifier(i) )
              out << ' ';

            out << *(leaves[i]);
          }
        }
        else{
          out << value;
        }

        break;
      }

      case (expType::type):{
        // [const] [int] [*]
        if(leafCount){
          out << *(leaves[0]);

          for(int i = 1; i < leafCount; ++i){
            if( !leaves[i - 1]->hasAnArrayQualifier() )
              out << ' ';

            out << *(leaves[i]);
          }

          if(leaves[leafCount - 1]->info & expType::type)
            out << ' ';
        }
        // [int]
        else{
          out << value;
        }

        break;
      }

      case (expType::type | expType::namespace_):{
        for(int i = 0; i < leafCount; ++i)
          out << *(leaves[i]);

        break;
      }

      case (expType::presetValue):{
        out << value;

        break;
      }

      case (expType::operator_):{
        out << value;

        break;
      }

      case (expType::unknown):{
        out << value;

        break;
      }

      case (expType::variable):{
        // [[[const] [int] [*]] [x]]
        if(leafCount){
          const bool hasLQualifier = (leaves[0]->info & (expType::qualifier |
                                                         expType::type));

          const bool hasRQualifier = (((hasLQualifier + 1) < leafCount) &&
                                      (leaves[hasLQualifier + 1]->info & (expType::qualifier |
                                                                          expType::type)));

          if(hasLQualifier){
            out << *(leaves[0]);

            if((leaves[0]->info & expType::qualifier) ||
               ((leaves[0]->leafCount) &&
                (leaves[0]->lastLeaf()->info & expType::qualifier))){

              out << ' ';
            }
          }

          out << *(leaves[hasLQualifier]);

          if(hasRQualifier){
            if( !(leaves[hasLQualifier + 1]->hasAnArrayQualifier()) )
              out << ' ';

            out << *(leaves[hasLQualifier + 1]);
          }
        }
        // [x]
        else{
          out << value;
        }

        break;
      }

      case (expType::function | expType::prototype):{
        if(leafCount)
          out << tab << getVarInfo(0) << ";\n";

        break;
      }

      case (expType::function | expType::declaration):{
        if(leafCount)
          out << tab << getVarInfo(0);

        break;
      }

      case (expType::function):{
        out << value;

        if(leafCount)
          out << *(leaves[0]);

        break;
      }

      case (expType::functionPointer):{
        out << *(leaves[0]) << " (*" << *(leaves[1]) << ")"
            << '(';

        expNode *argNode = leaves[2];

        if(argNode->leafCount){
          for(int i = 0; i < (argNode->leafCount - 1); ++i)
            out << *(argNode->leaves[i]) << ", ";

          out << *(argNode->leaves[argNode->leafCount - 1]);
        }

        out << ')';

        break;
      }

      case (expType::declaration):{
        if(leafCount){
          out << tab << leaves[0]->toString();

          for(int i = 1; i < leafCount; ++i)
            out << ", " << leaves[i]->toString();

          if( !(flags & expFlag::noSemicolon) )
            out << ';';

          if( !(flags & expFlag::noNewline) )
            out << '\n';
        }

        break;
      }

      case (expType::struct_):{
        if(leafCount){
          typeInfo &type = *((typeInfo*) leaves[0]->leaves[0]);
          out << type.toString(tab) << ";\n";
        }

        break;
      }

      case (expType::varInfo | expType::type):{
        out << getVarInfo();

        break;
      }

      case (expType::varInfo):{
        out << getVarInfo().toString(false);

        break;
      }

      case (expType::typeInfo):{
        out << getTypeInfo().toString(tab) << ";\n";

        break;
      }

      case (expType::C | expType::cast_):{
        out << '(';

        for(int i = 0; i < leafCount; ++i)
          out << *(leaves[i]);

        out << ')';

        break;
      }

      case (expType::namespace_):{
        break;
      }

      case (expType::macro_):{
        out << tab << value << '\n';
        break;
      }

      case (expType::goto_):{
        out << tab << "goto " << value << ';';
        break;
      }

      case (expType::gotoLabel_):{
        out << tab << value << ':';
        break;
      }

      case (expType::case_):{
        out << tab << "case ";

        for(int i = 0; i < leafCount; ++i)
          out << *(leaves[i]);

        if((up == NULL) &&
           (sInfo->type & simpleStatementType))
          out << ';';

        break;
      }

      case (expType::return_):{
        out << tab;

        for(int i = 0; i < leafCount; ++i)
          out << *(leaves[i]);

        out << '\n';

        break;
      }

      case (expType::occaFor):{
        out << value << ' ';
        break;
      }

      case (expType::checkSInfo):{
        if(sInfo->type & flowStatementType){
          out << tab;

          if(sInfo->type & forStatementType)
            out << "for(";
          else if(sInfo->type & whileStatementType)
            out << "while(";
          else if(sInfo->type & ifStatementType){
            if(sInfo->type == ifStatementType)
              out << "if(";
            else if(sInfo->type == elseIfStatementType)
              out << "else if(";
            else
              out << "else";
          }
          else if(sInfo->type & switchStatementType)
            out << "switch(";

          if(leafCount){
            if(leaves[0]->info & expType::declaration)
              leaves[0]->printOn(out, "", (expFlag::noNewline |
                                           expFlag::noSemicolon));
            else
              out << *(leaves[0]);

            for(int i = 1; i < leafCount; ++i)
              out << "; " << *(leaves[i]);
          }

          if( !(sInfo->type & gotoStatementType) &&
              (sInfo->type != elseStatementType) ){
            out << ")";
          }
          else if(sInfo->type & gotoStatementType){
            out << ":";
          }
        }
      }

      case (expType::printValue):{
        out << value << ' ';

        break;
      }

      case (expType::printLeaves):{
        if(leafCount){
          for(int i = 0; i < leafCount; ++i)
            out << leaves[i]->value << ' ';

          out << '\n';
        }

        break;
      }

      default:{
        if(info & expType::typedef_){
          const int oldInfo = info;

          out << "typedef ";

          info &= ~expType::typedef_;

          out << *this;

          info = oldInfo;

          if(info & expType::struct_)
            out << ";\n";
        }
      }
      };
    }

    std::string expNode::toString(const std::string &tab){
      std::stringstream ss;

      printOn(ss, tab);

      return ss.str();
    }

    expNode::operator std::string (){
      return toString();
    }

    std::ostream& operator << (std::ostream &out, expNode &n){
      n.printOn(out);

      return out;
    }
    //==============================================


    //---[ Statement Functions ]--------------------
    statement::statement(parserBase &pb) :
      depth(-1),
      type(blockStatementType),

      up(NULL),

      varOriginMap(pb.varOriginMap),
      varUsedMap(pb.varUsedMap),

      expRoot(*this),

      statementCount(0),
      statementStart(NULL),
      statementEnd(NULL) {}

    statement::statement(const int depth_,
                         varOriginMap_t &varOriginMap_,
                         varUsedMap_t &varUsedMap_) :
      depth(depth_),
      type(blockStatementType),

      up(NULL),

      varOriginMap(varOriginMap_),
      varUsedMap(varUsedMap_),

      expRoot(*this),

      statementCount(0),
      statementStart(NULL),
      statementEnd(NULL) {}

    statement::statement(const int depth_,
                         const int type_,
                         statement *up_) :
      depth(depth_),
      type(type_),

      up(up_),

      varOriginMap(up_->varOriginMap),
      varUsedMap(up_->varUsedMap),

      expRoot(*this),

      statementCount(0),
      statementStart(NULL),
      statementEnd(NULL) {}

    statement::~statement(){};

    statement* statement::makeSubStatement(){
      return new statement(depth + 1,
                           0, this);
    }

    std::string statement::getTab() const {
      std::string ret = "";

      for(int i = 0; i < depth; ++i)
        ret += "  ";

      return ret;
    }

    void statement::labelStatement(strNode *&nodeRoot, expNode *expPtr){
      type = findStatementType(nodeRoot, expPtr);
    }

    int statement::findStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(nodeRoot->type == macroKeywordType)
        return checkMacroStatementType(nodeRoot, expPtr);

      else if(nodeRoot->type == 0)
        return 0;

      else if(nodeRoot->type == keywordType["occaOuterFor0"])
        return checkOccaForStatementType(nodeRoot, expPtr);

      else if((nodeRoot->type & typedefType) |
              (nodeRoot->type & structType))
        return checkStructStatementType(nodeRoot, expPtr);

      else if(nodeRoot->type & (operatorType |
                                presetValue))
        return checkUpdateStatementType(nodeRoot, expPtr);

      else if(nodeHasDescriptor(nodeRoot))
        return checkDescriptorStatementType(nodeRoot, expPtr);

      else if(nodeRoot->type & unknownVariable){
        if(nodeRoot->right &&
           nodeRoot->right->value == ":")
          return checkGotoStatementType(nodeRoot, expPtr);

        return checkUpdateStatementType(nodeRoot, expPtr);
      }

      else if(nodeRoot->type & flowControlType)
        return checkFlowStatementType(nodeRoot, expPtr);

      else if(nodeRoot->type & specialKeywordType)
        return checkSpecialStatementType(nodeRoot, expPtr);

      else if(nodeRoot->type & brace)
        return checkBlockStatementType(nodeRoot, expPtr);

      // Statement: (int) 3;
      else if(nodeRoot->type & parentheses)
        return checkUpdateStatementType(nodeRoot, expPtr);

      // Statement: [;]
      else if(nodeRoot->type & endStatement)
        return checkUpdateStatementType(nodeRoot, expPtr);

      else {
        while(nodeRoot &&
              !(nodeRoot->type & endStatement))
          nodeRoot = nodeRoot->right;

        return updateStatementType;
      }
    }

    int statement::checkMacroStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(expPtr){
        expPtr->info  = expType::macro_;
        expPtr->value = nodeRoot->value;
      }

      return macroStatementType;
    }

    int statement::checkOccaForStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(expPtr){
        expPtr->info  = expType::occaFor;
        expPtr->value = nodeRoot->value;
      }

      nodeRoot = nodeRoot->right;

      return keywordType["occaOuterFor0"];
    }

    int statement::checkStructStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(!typeInfo::statementIsATypeInfo(*this, nodeRoot))
        return checkDescriptorStatementType(nodeRoot);

      while(nodeRoot){
        if(nodeRoot->type & endStatement)
          break;

        nodeRoot = nodeRoot->right;
      }

      return structStatementType;
    }

    int statement::checkUpdateStatementType(strNode *&nodeRoot, expNode *expPtr){
      while(nodeRoot){
        if(nodeRoot->type & endStatement)
          break;

        nodeRoot = nodeRoot->right;
      }

      return updateStatementType;
    }

    int statement::checkDescriptorStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(typeInfo::statementIsATypeInfo(*this, nodeRoot))
        return checkStructStatementType(nodeRoot);

      varInfo var;
      nodeRoot = var.loadFrom(*this, nodeRoot);

      if( !(var.info & varType::functionDef) ){
        while(nodeRoot){
          if(nodeRoot->type & endStatement)
            break;

          nodeRoot = nodeRoot->right;
        }
      }

      if(var.info & varType::var)
        return declareStatementType;
      else if(var.info & varType::functionDec)
        return functionPrototypeType;
      else
        return functionDefinitionType;
    }

    int statement::checkGotoStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(expPtr){
        expPtr->info  = expType::gotoLabel_;
        expPtr->value = nodeRoot->value;
      }

      nodeRoot = nodeRoot->right;

      return gotoStatementType;
    }

    int statement::checkFlowStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(expPtr)
        expPtr->info  = expType::checkSInfo;

      std::string &nodeValue = nodeRoot->value;

      nodeRoot = nodeRoot->right;

      if((nodeValue != "else") &&
         (nodeValue != "do")){

        nodeRoot = nodeRoot->right;
      }

      if(nodeValue == "for")
        return forStatementType;
      else if(nodeValue == "while")
        return whileStatementType;
      else if(nodeValue == "do")
        return doWhileStatementType;
      else if(nodeValue == "if")
        return ifStatementType;
      else if(nodeValue == "else if")
        return elseIfStatementType;
      else if(nodeValue == "else")
        return elseStatementType;
      else if(nodeValue == "switch")
        return switchStatementType;

      std::cout << "You found the [Waldo 2] error in:\n"
                << prettyString(nodeRoot, "  ");
      throw 1;

      return 0;
    }

    int statement::checkSpecialStatementType(strNode *&nodeRoot, expNode *expPtr){
      while(nodeRoot){
        if(nodeRoot->type & endStatement)
          break;

        nodeRoot = nodeRoot->right;
      }

      return blankStatementType;
    }

    int statement::checkBlockStatementType(strNode *&nodeRoot, expNode *expPtr){
      nodeRoot = lastNode(nodeRoot);

      return blockStatementType;
    }

    void statement::addType(typeInfo &type){
      scopeTypeMap[type.name] = &type;
    }

    void statement::addTypedef(const std::string &typedefName){
      typeInfo &type = *(new typeInfo);
      type.name = typedefName;
      scopeTypeMap[typedefName] = &type;
    }

    bool statement::nodeHasQualifier(strNode *n) const {
      if( !(n->type & qualifierType) )
        return false;

      // short and long can be both:
      //    specifiers and qualifiers
      if(n->type == keywordType["long"]){
        if((n->right) &&
           (hasTypeInScope(n->right->value))){

          return true;
        }
        else
          return false;
      }

      return true;
    }

    bool statement::nodeHasSpecifier(strNode *n) const {
      return ((n->type & specifierType) ||
              ((n->type & unknownVariable) &&
               ( hasTypeInScope(n->value) )));
    }

    bool statement::nodeHasDescriptor(strNode *n) const {
      if(nodeHasSpecifier(n) || nodeHasQualifier(n))
        return true;

      return false;
    }

    varInfo statement::loadVarInfo(strNode *&nodePos){
      varInfo info;
      return info;
#if 0
#if 0
      while(nodePos &&
            ((nodePos->type & qualifierType) ||
             nodeHasSpecifier(nodePos))){

        if(nodePos->type & structType){
          info.descriptors.push_back(*nodePos);
          info.typeInfo |= structType;
        }

        else if(nodeHasQualifier(nodePos)){
          if(nodePos->value == "*"){
            info.typeInfo |= heapPointerType;
            ++info.pointerCount;

            if(nodePos->right &&
               nodePos->right->value == "const"){
              info.typeInfo |= constPointerType;
              nodePos = nodePos->right;
            }
          }
          else if(nodePos->value == "&")
            info.typeInfo |= referenceType;
          else{
            if((nodePos->value  == "texture")              ||
               ((nodePos->value == "image1d_t")            ||
                (nodePos->value == "image2d_t"))           ||
               (nodePos->value  == "cudaSurfaceObject_t"))
              info.typeInfo |= textureType;

            info.descriptors.push_back(nodePos->value);
          }
        }

        else if(nodeHasSpecifier(nodePos)){
          info.type = hasTypeInScope(*nodePos);

          if(info.type == NULL){
            std::cout << "Type [" << *nodePos << "] is not defined.\nFound in:\n";
            nodePos->print();
            throw 1;
          }
        }

        if(nodePos->down.size() != 0)
          break;

        nodePos = nodePos->right;
      }

      if((nodePos == NULL) ||
         (nodePos->type & (endSection |
                           endStatement))){

        if(nodePos &&
           nodePos->value == ":"){

          info.bitField = atoi(nodePos->right->value.c_str());
          nodePos = nodePos->right->right;
        }

        return info;
      }

      const int downCount = nodePos->down.size();

      if(downCount == 0){
        info.name      = nodePos->value;
        info.typeInfo |= variableType;

        nodePos = nodePos->right;

        if(nodePos &&
           nodePos->value == ":"){

          info.bitField = atoi(nodePos->right->value.c_str());
          nodePos = nodePos->right->right;

          return info;
        }
      }
      else{
        strNode *downNode = nodePos->down[0];

        if(downNode->type == startParentheses){
          // [-] Only for C
          if((2 <= downCount) &&
             (nodePos->down[1]->type == startParentheses)){

            downNode = downNode->right;

            while(downNode->value == "*"){
              ++(info.pointerCount);
              downNode = downNode->right;
            }

            info.name      = downNode->value;
            info.typeInfo |= functionPointerType;

            if(downNode->down.size()){
              const int downCount2 = downNode->down.size();

              for(int i = 0; i < downCount2; ++i){
                if(downNode->down[i]->type != startBracket)
                  break;

                std::string sps = prettyString(downNode->down[i]);
                sps = sps.substr(1, sps.size() - 2); // Remove '[' and ']'

                info.stackPointerSizes.push_back(sps);
              }
            }

            downNode = nodePos->down[1]->right;

            while(downNode){
              varInfo &arg = *(new varInfo);
              arg = loadVarInfo(downNode);

              info.vars.push_back(&arg);

              // Loaded last arg
              if((downNode == NULL) ||
                 (downNode->right == NULL))
                break;

              downNode = downNode->right;
            }

            nodePos = nodePos->right;
          }
          else{
            info.name      = nodePos->value;
            info.typeInfo |= functionType;

            downNode = downNode->right;

            while(downNode){
              varInfo &arg = *(new varInfo);
              arg = loadVarInfo(downNode);

              info.vars.push_back(&arg);

              downNode = downNode->right;

              // Loaded last arg
              if(downNode == NULL)
                break;
            }

            // Distinguish between prototypes and function calls
            if(nodePos->right &&
               (nodePos->right->value == ";")){
              if(info.type)
                info.typeInfo |= protoType;
              else
                info.typeInfo |= functionCallType;
            }
          }
        }
        else if(downNode->type == startBracket){
          info.name      = nodePos->value;
          info.typeInfo |= (variableType | stackPointerType);

          for(int i = 0; i < downCount; ++i){
            if(nodePos->down[i]->type != startBracket)
              break;

            std::string sps = prettyString(nodePos->down[i]);
            sps = sps.substr(1, sps.size() - 2); // Remove '[' and ']'

            info.stackPointerSizes.push_back(sps);
          }
        }
      }

      return info;
#else
      while(nodePos                         &&
            !(nodePos->type & presetValue)  &&
            !(nodePos->type & endStatement) && // For bitfields
            (!(nodePos->type & unknownVariable) ||
             hasTypeInScope(nodePos->value))){

        if(nodePos->type & structType){
          info.descriptors.push_back(*nodePos);
          info.typeInfo |= structType;
        }

        else if(nodeHasQualifier(nodePos)){
          if(nodePos->value == "*"){
            info.typeInfo |= heapPointerType;
            ++info.pointerCount;

            if(nodePos->right &&
               nodePos->right->value == "const"){
              info.typeInfo |= constPointerType;
              nodePos = nodePos->right;
            }
          }
          else if(nodePos->value == "&")
            info.typeInfo |= referenceType;
          else{
            if((nodePos->value  == "texture")              ||
               ((nodePos->value == "image1d_t")            ||
                (nodePos->value == "image2d_t"))           ||
               (nodePos->value  == "cudaSurfaceObject_t"))
              info.typeInfo |= textureType;

            info.descriptors.push_back(*nodePos);
          }
        }

        else if(nodeHasSpecifier(nodePos)){
          info.type = hasTypeInScope(*nodePos);

          if(info.type == NULL){
            std::cout << "Type [" << *nodePos << "] is not defined.\nFound in:\n";
            nodePos->print();
            throw 1;
          }
        }

        nodePos = nodePos->right;
      }

      if((nodePos == NULL) ||
         (nodePos->type & endStatement)) // For bitfields
        return info;

      info.name = *nodePos;

      const int downCount = nodePos->down.size();

      if(downCount){
        if(nodePos->down[0]->type == startParentheses){
          strNode *argPos = nodePos->down[0];
          info.typeInfo |= functionType;

          strNode *lastPos = lastNode(nodePos);

          // Distinguish between prototypes and function calls
          if(lastPos->value == ";"){
            if(info.type)
              info.typeInfo |= protoType;
            else
              info.typeInfo |= functionCallType;
          }
        }
        else if(nodePos->down[0]->type == startBracket){
          info.typeInfo |= (variableType | stackPointerType);

          for(int i = 0; i < downCount; ++i){
            nodePos->down[i]->type = startBracket;

            std::string sps = prettyString(nodePos->down[i]);
            sps = sps.substr(1, sps.size() - 2); // Remove '[' and ']'

            info.stackPointerSizes.push_back(sps);
          }
        }
      }
      else{
        info.typeInfo |= variableType;
        nodePos = nodePos->right;
      }

      return info;
#endif
#endif
    }

    typeInfo* statement::hasTypeInScope(const std::string &typeName) const {
      cScopeTypeMapIterator it = scopeTypeMap.find(typeName);

      if(it != scopeTypeMap.end())
        return it->second;

      if(up)
        return up->hasTypeInScope(typeName);

      return NULL;
    }

    varInfo* statement::hasVariableInScope(const std::string &varName) const {
      const statement *sPos = this;

      while(sPos){
        cScopeVarMapIterator it = sPos->scopeVarMap.find(varName);

        if(it != sPos->scopeVarMap.end())
          return it->second;

        sPos = sPos->up;
      }

      return NULL;
    }

    bool statement::hasDescriptorVariable(const std::string descriptor) const {
      return hasQualifier(descriptor);
    }

    bool statement::hasDescriptorVariableInScope(const std::string descriptor) const {
      if(hasDescriptorVariable(descriptor))
        return true;

      if(up != NULL)
        return up->hasDescriptorVariable(descriptor);

      return false;
    }

    //---[ Loading ]--------------------
    void statement::loadAllFromNode(strNode *nodeRoot){
      while(nodeRoot)
        nodeRoot = loadFromNode(nodeRoot);
    }

    // [+]
    strNode* statement::loadFromNode(strNode *nodeRoot){
      statement *newStatement = makeSubStatement();
      strNode * nodeRootEnd   = nodeRoot;

      newStatement->expRoot.loadFromNode(nodeRootEnd);
      const int st = newStatement->type;

      if(st & invalidStatementType){
        std::cout << "Not a valid statement\n";
        throw 1;
      }

      addStatement(newStatement);

      if(st & simpleStatementType){
        nodeRootEnd = newStatement->loadSimpleFromNode(st,
                                                       nodeRoot,
                                                       nodeRootEnd);
      }

      else if(st & flowStatementType){
        if(st & forStatementType)
          nodeRootEnd = newStatement->loadForFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd);

        else if(st & whileStatementType)
          nodeRootEnd = newStatement->loadWhileFromNode(st,
                                                        nodeRoot,
                                                        nodeRootEnd);

        else if(st & ifStatementType)
          nodeRootEnd = loadIfFromNode(st,
                                       nodeRoot,
                                       nodeRootEnd);

        else if(st & switchStatementType)
          nodeRootEnd = newStatement->loadSwitchFromNode(st,
                                                         nodeRoot,
                                                         nodeRootEnd);

        else if(st & gotoStatementType)
          nodeRootEnd = newStatement->loadGotoFromNode(st,
                                                       nodeRoot,
                                                       nodeRootEnd);
      }
      else if(st & blockStatementType)
        nodeRootEnd = newStatement->loadBlockFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd);

      else if(st & functionStatementType){
        if(st & functionDefinitionType)
          nodeRootEnd = newStatement->loadFunctionDefinitionFromNode(st,
                                                                     nodeRoot,
                                                                     nodeRootEnd);

        else if(st & functionPrototypeType)
          nodeRootEnd = newStatement->loadFunctionPrototypeFromNode(st,
                                                                    nodeRoot,
                                                                    nodeRootEnd);
      }

      else if(st & structStatementType)
        nodeRootEnd = newStatement->loadStructFromNode(st,
                                                       nodeRoot,
                                                       nodeRootEnd);

      else if(st & blankStatementType)
        nodeRootEnd = newStatement->loadBlankFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd);

      else if(st & macroStatementType)
        nodeRootEnd = newStatement->loadMacroFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd);

      return nodeRootEnd;
    }

    void statement::setExpNodeFromStrNode(expNode &exp,
                                          strNode *nodePos){
      expNode *tmp = createExpNodeFrom(nodePos);

      exp.sInfo = this;

      exp.value = tmp->value;
      exp.info  = tmp->info;

      exp.leafCount = tmp->leafCount;
      exp.leaves    = tmp->leaves;

      delete tmp;
    }

    expNode* statement::createExpNodeFrom(strNode *nodeRoot){
      loadFromNode(nodeRoot);

      statementNode *sn = statementEnd;

      if(statementStart == statementEnd)
        statementStart = statementEnd->left;

      statementEnd = statementEnd->left;

      if(statementEnd)
        statementEnd->right = NULL;

      --(statementCount);

      expNode &ret = *(sn->value->expRoot.clone(*this));

      delete sn->value;
      delete sn;

      return &ret;
    }

    expNode* statement::createExpNodeFrom(const std::string &source){
      strNode *nodeRoot = parserNamespace::splitContent(source);
      nodeRoot          = parserNamespace::labelCode(nodeRoot);

      expNode *ret = createExpNodeFrom(nodeRoot);

      free(nodeRoot);

      return ret;
    }

    strNode* statement::loadSimpleFromNode(const int st,
                                           strNode *nodeRoot,
                                           strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    strNode* statement::loadOneStatementFromNode(const int st,
                                                 strNode *nodeRoot,
                                                 strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;

      if(nodeRootEnd){
        if(nodeRootEnd->type == startBrace)
          loadAllFromNode(nodeRootEnd->down);
        else
          return loadFromNode(nodeRootEnd);
      }

      return nextNode;
    }

    strNode* statement::loadForFromNode(const int st,
                                        strNode *nodeRoot,
                                        strNode *nodeRootEnd){

      return loadOneStatementFromNode(st, nodeRoot, nodeRootEnd);
    }

    strNode* statement::loadWhileFromNode(const int st,
                                          strNode *nodeRoot,
                                          strNode *nodeRootEnd){
      if(st == whileStatementType)
        return loadOneStatementFromNode(st, nodeRoot, nodeRootEnd);
      else{
        strNode *nextNode = loadOneStatementFromNode(st, nodeRootEnd, nodeRootEnd);
        type = whileStatementType;

        expRoot.loadFromNode(nextNode);

        type = doWhileStatementType;

        // Skip the [;] after [while()]
        if(nextNode &&
           (nextNode->value == ";")){

          nextNode = nextNode->right;
        }

        return nextNode;
      }
    }

    strNode* statement::loadIfFromNode(const int st_,
                                       strNode *nodeRoot,
                                       strNode *nodeRootEnd){
      statement *newStatement = statementEnd->value;

      strNode *nextNode = newStatement->loadOneStatementFromNode(st_,
                                                                 nodeRoot,
                                                                 nodeRootEnd);

      if(nextNode == NULL)
        return NULL;

      nodeRoot    = nextNode;
      nodeRootEnd = nextNode;

      int st      = findStatementType(nodeRootEnd);
      int stCheck = elseIfStatementType;

      nodeRootEnd = nextNode;

      while(true){
        if(st != stCheck){
          if(stCheck == elseIfStatementType)
            stCheck = elseStatementType;
          else
            break;
        }
        else if(nextNode == NULL){
          break;
        }
        else{
          newStatement = makeSubStatement();
          newStatement->expRoot.loadFromNode(nodeRootEnd);

          if(st & invalidStatementType){
            std::cout << "Not a valid statement\n";
            throw 1;
          }

          addStatement(newStatement);

          nextNode = newStatement->loadOneStatementFromNode(st,
                                                            nodeRoot,
                                                            nodeRootEnd);

          nodeRoot    = nextNode;
          nodeRootEnd = nextNode;

          if(nodeRootEnd){
            st = findStatementType(nodeRootEnd);

            nodeRootEnd = nextNode;
          }
        }
      }

      return nextNode;
    }

    // [-] Missing
    strNode* statement::loadSwitchFromNode(const int st,
                                           strNode *nodeRoot,
                                           strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    strNode* statement::loadGotoFromNode(const int st,
                                         strNode *nodeRoot,
                                         strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    strNode* statement::loadFunctionDefinitionFromNode(const int st,
                                                       strNode *nodeRoot,
                                                       strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      if(nodeRootEnd)
        loadAllFromNode(nodeRootEnd->down);

      return nextNode;
    }

    strNode* statement::loadFunctionPrototypeFromNode(const int st,
                                                      strNode *nodeRoot,
                                                      strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    strNode* statement::loadBlockFromNode(const int st,
                                          strNode *nodeRoot,
                                          strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot->down)
        loadAllFromNode(nodeRoot->down);

      return nextNode;
    }

    strNode* statement::loadStructFromNode(const int st,
                                           strNode *nodeRoot,
                                           strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    // [-] Missing
    strNode* statement::loadBlankFromNode(const int st,
                                          strNode *nodeRoot,
                                          strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    // [-] Missing
    strNode* statement::loadMacroFromNode(const int st,
                                          strNode *nodeRoot,
                                          strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }
    //==================================


    statementNode* statement::getStatementNode(){
      if(up != NULL){
        statementNode *ret = up->statementStart;

        while(ret){
          if(ret->value == this)
            return ret;

          ret = ret->right;
        }
      }

      return NULL;
    }

    statement& statement::pushNewStatementLeft(const int type_){
      statementNode *newSN = new statementNode(up->makeSubStatement());

      statement *newS = newSN->value;
      newS->type      = type_;

      statementNode *sn = getStatementNode();

      if(up->statementStart == sn)
        up->statementStart = newSN;

      if(sn->left)
        sn->left->right = newSN;

      newSN->left  = sn->left;
      newSN->right = sn;

      sn->left = newSN;

      return *newS;
    }

    statement& statement::pushNewStatementRight(const int type_){
      statementNode *newSN = new statementNode(up->makeSubStatement());

      statement *newS = newSN->value;
      newS->type      = type_;

      statementNode *sn = getStatementNode();

      if(up->statementEnd == sn)
        up->statementEnd = newSN;

      if(sn->right)
        sn->right->left = newSN;

      newSN->right  = sn->right;
      newSN->left   = sn;

      sn->right = newSN;

      return *newS;
    }

    void statement::pushLeftFromSource(statementNode *target,
                                       const std::string &source){
      loadFromNode(labelCode( splitContent(source) ));

      statementNode *newSN = statementEnd;

      statementEnd = statementEnd->left;

      if(statementEnd)
        statementEnd->right = NULL;

      if(statementStart == target)
        statementStart = newSN;

      if(target->left)
        target->left->right = newSN;

      newSN->left  = target->left;
      newSN->right = target;

      target->left = newSN;
    }

    void statement::pushRightFromSource(statementNode *target,
                                        const std::string &source){
      loadFromNode(labelCode( splitContent(source) ));

      statementNode *newSN = statementEnd;

      statementEnd = statementEnd->left;

      if(statementEnd)
        statementEnd->right = NULL;

      if(statementEnd == target)
        statementEnd = newSN;

      if(target->right)
        target->right->left = newSN;

      newSN->right  = target->right;
      newSN->left   = target;

      target->right = newSN;
    }

    void statement::checkIfVariableIsDefined(varInfo &var,
                                             statement *origin){
      scopeVarMapIterator it = scopeVarMap.find(var.name);

      if(it != scopeVarMap.end()      &&
         !var.hasQualifier("extern") &&
         !((var.info & varType::functionDef))){

        std::cout << "Variable [" << var.name << "] defined in:\n"
                  << *origin
                  << "is already defined in:\n"
                  << *this;
        throw 1;
      }
    }

    varInfo& statement::addVariable(varInfo &var,
                                    statement *origin){
      varInfo &newVar = *(new varInfo);
      newVar = var.clone();

      addVariable(&newVar, origin);

      return newVar;
    }

    void statement::addVariable(varInfo *var,
                                statement *origin){
      checkIfVariableIsDefined(*var, origin);

      scopeVarMap[var->name] = var;

      if(origin == NULL)
        varOriginMap[var] = this;
      else
        varOriginMap[var] = origin;
    }

    void statement::addStatement(statement *newStatement){
      if(statementStart != NULL){
        ++statementCount;
        statementEnd = statementEnd->push(newStatement);
      }
      else{
        statementCount = 1;
        statementStart = new node<statement*>(newStatement);
        statementEnd   = statementStart;
      }
    }

    statement* statement::clone(){
      statement *newStatement;

      if(up){
        newStatement = new statement(depth,
                                     type, up);
      }
      else {
        newStatement = new statement(depth,
                                     varOriginMap,
                                     varUsedMap);
      }

      expRoot.cloneTo(newStatement->expRoot);

      newStatement->scopeVarMap = scopeVarMap;

      newStatement->statementCount = statementCount;

      newStatement->statementStart = NULL;
      newStatement->statementEnd   = NULL;

      if(statementCount == 0)
        return newStatement;

      statementNode *nodePos = statementStart;

      // [-] Broken
      // for(int i = 0; i < statementCount; ++i){
      //   newStatement->addStatement( nodePos->value->clone() );
      //   nodePos = nodePos->right;
      // }

      while(nodePos){
        newStatement->addStatement( nodePos->value->clone() );
        nodePos = nodePos->right;
      }

      return newStatement;
    }

    void statement::printVariablesInStatement(){
      scopeVarMapIterator it = scopeVarMap.begin();

      while(it != scopeVarMap.end()){
        std::cout << "  " << *(it->second) << '\n';

        ++it;
      }
    }

    void statement::printVariablesInScope(){
      if(up)
        up->printVariablesInScope();

      printVariablesInStatement();
    }

    void statement::printTypesInScope(){
      if(up)
        up->printTypesInScope();

      printTypesInStatement();
    }

    void statement::printTypesInStatement(){
      scopeTypeMapIterator it = scopeTypeMap.begin();

      while(it != scopeTypeMap.end()){
        std::cout << (it->first) << '\n';

        ++it;
      }
    }

    void statement::printTypeDefsInStatement(){
      scopeTypeMapIterator it = scopeTypeMap.begin();

      while(it != scopeTypeMap.end()){
        std::cout << (it->second)->toString("  ") << '\n';

        ++it;
      }
    }

    //---[ Statement Info ]-----------
    void statement::swapExpWith(statement &s){
      expNode::swap(expRoot, s.expRoot);
    }

    bool statement::hasQualifier(const std::string &qualifier) const {
      if(type & declareStatementType){
        const varInfo &var = cGetDeclarationVarInfo(0);
        return var.hasQualifier(qualifier);
      }
      else if(type & functionStatementType){
        const varInfo &var = expRoot.cGetVarInfo(0);
        return var.hasQualifier(qualifier);
      }
      else if(type & forStatementType){
        if(expRoot.leafCount){
          expNode &node1 = *(expRoot.leaves[0]);

          if((node1.leafCount) &&
             (node1.leaves[0]->info & expType::type)){

            return node1.leaves[0]->hasQualifier(qualifier);
          }
        }

        return false;
      }

      return false;
    }

    void statement::addQualifier(const std::string &qualifier,
                                 const int pos){
      if(hasQualifier(qualifier))
        return;

      if(type & declareStatementType){
        varInfo &var = getDeclarationVarInfo(0);
        var.addQualifier(qualifier);
      }
      else if(type & functionStatementType){
        varInfo &var = expRoot.getVarInfo(0);
        var.addQualifier(qualifier, pos);
      }
      else if(type & forStatementType){
        if(expRoot.leafCount){
          expNode &node1    = *(expRoot.leaves[0]);
          expNode &qualNode = *(node1.leaves[0]);

          if( !(qualNode.leaves[0]->info & expType::qualifier) )
            qualNode.addNode(expType::qualifier, 0);

          qualNode.leaves[0]->value = qualifier;
        }
      }
    }

    void statement::removeQualifier(const std::string &qualifier){
      if(!hasQualifier(qualifier))
        return;

      if(type & declareStatementType){
        varInfo &var = getDeclarationVarInfo(0);
        var.removeQualifier(qualifier);
      }
      else if(type & functionStatementType){
      }
      else if(type & forStatementType){
      }
    }

    varInfo& statement::getDeclarationVarInfo(const int pos){
      expNode *varNode = expRoot.getVariableInfoNode(pos);
      return varNode->getVarInfo();
    }

    const varInfo& statement::cGetDeclarationVarInfo(const int pos) const {
      expNode *varNode = expRoot.getVariableInfoNode(pos);
      return varNode->cGetVarInfo();
    }

    expNode* statement::getDeclarationVarNode(const int pos){
      if(type & declareStatementType)
        return expRoot.leaves[pos];

      return NULL;
    }

    std::string statement::getDeclarationVarName(const int pos){
      if(type & declareStatementType){
        varInfo &var = getDeclarationVarInfo(pos);
        return var.name;
      }

      return "";
    }

    expNode* statement::getDeclarationVarInitNode(const int pos){
      if(type & declareStatementType)
        return expRoot.getVariableInitNode(pos);

      return NULL;
    }

    int statement::getDeclarationVarCount() const {
      if(type & declareStatementType)
        return expRoot.leafCount;

      return 0;
    }

    varInfo* statement::getFunctionVar(){
      if(type & functionStatementType){
        return &(expRoot.getVarInfo(0));
      }

      printf("Not added yet\n");
      throw 1;

      return NULL;
    }

    std::string statement::getFunctionName(){
      if(type & functionStatementType){
        return getFunctionVar()->name;
      }

      printf("Not added yet\n");
      throw 1;

      return "";
    }

    void statement::setFunctionName(const std::string &newName){
      if(type & functionStatementType){
        getFunctionVar()->name = newName;
        return;
      }

      printf("Not added yet\n");
      throw 1;
    }

    bool statement::functionHasQualifier(const std::string &qName){
      if(type & functionStatementType){
        return getFunctionVar()->hasQualifier(qName);
      }

      printf("Not added yet\n");
      throw 1;
    }

    int statement::getFunctionArgCount(){
      if(type & functionStatementType){
        return getFunctionVar()->argumentCount;
      }

      return 0;
    }

    std::string statement::getFunctionArgType(const int pos){
      if(type & functionDefinitionType){
        return getFunctionVar()->baseType->name;
      }

      return "";
    }

    std::string statement::getFunctionArgName(const int pos){
      if(type & functionDefinitionType){
        return getFunctionVar()->argumentVarInfos[pos].name;
      }

      return "";
    }

    varInfo* statement::getFunctionArgVar(const int pos){
      if(type & functionDefinitionType){
        return &(getFunctionVar()->argumentVarInfos[pos]);
      }

      return NULL;
    }

    void statement::addFunctionArg(const int pos, varInfo &var){
      if( !(type & functionStatementType) )
        return;

      getFunctionVar()->addArgument(pos, var);
    }

    int statement::getForStatementCount() const {
      if(type & forStatementType)
        return expRoot.leafCount;

      return 0;
    }
    //================================

    // autoMode: Handles newlines and tabs
    std::string statement::prettyString(strNode *nodeRoot,
                                        const std::string &tab_,
                                        const bool autoMode) const {
      return "";
#if 0
      strNode *nodePos = nodeRoot;

      std::string tab = tab_;
      std::string ret = "";

      while(nodePos){
        if(nodePos->type & operatorType){

          if(nodePos->type & binaryOperatorType){

            // char *blah
            if(nodeHasQualifier(nodePos)){

              // [char ][*][blah]
              // or
              // [int ][a][ = ][0][, ][*][b][ = ][1][;]
              //                       ^
              if(nodePos->left &&
                 ((nodePos->left->type & descriptorType) ||
                  (nodePos->left->value == ","))){
                ret += *nodePos;

                // [const ][*][ const]
                if(nodePos->right &&
                   (nodePos->right->type & descriptorType) &&
                   !(nodePos->right->value == "*"))
                  ret += ' ';
              }
              else{
                ret += " ";
                ret += *nodePos;
                ret += " ";
              }
            }
            // [+] and [-]
            else if(nodePos->type & unitaryOperatorType){
              // (-blah ... )
              if(nodePos->left &&
                 !(nodePos->left->type & (presetValue | unknownVariable)) )
                ret += *nodePos;
              // a - b
              else{
                ret += " ";
                ret += *nodePos;
                ret += " ";
              }
            }
            else if(nodePos->value == ","){
              ret += ", ";
            }
            else if((nodePos->value == ".") || (nodePos->value == "::")){
              if(((nodePos->left == NULL) ||
                  !(nodePos->left->type & unknownVariable)) ||
                 ((nodePos->right == NULL) ||
                  !(nodePos->right->type & unknownVariable))){
                if(nodePos->left){
                  nodePos->up->print();
                  std::cout << "1. Error on:\n";
                  nodePos->left->print("  ");
                }
                else{
                  std::cout << "2. Error on:\n";
                  nodePos->print("  ");
                }

                throw 1;
              }

              ret += *nodePos;
            }
            else{
              ret += " ";
              ret += *nodePos;
              ret += " ";
            }

          }
          // [++] and [--]
          else if(nodePos->type & unitaryOperatorType){
            ret += *nodePos;
          }
          else if(nodePos->type & ternaryOperatorType){
            ret += " ? ";

            nodePos = nodePos->right;

            if((nodePos->right) == NULL){
              std::cout << "3. Error on:\n";
              nodePos->left->print("  ");
              throw 1;
            }

            if((nodePos->down).size())
              ret += prettyString(nodePos, "", autoMode);
            else
              ret += *nodePos;

            ret += " : ";

            if(((nodePos->right) == NULL)       ||
               ((nodePos->right->right) == NULL)){
              std::cout << "4. Error on:\n";
              nodePos->left->left->print("  ");
              throw 1;
            }

            nodePos = nodePos->right->right;

            if((nodePos->down).size())
              ret += prettyString(nodePos, "", autoMode);
            else
              ret += *nodePos;
          }
        }

        else if(nodePos->type & brace){
          if(nodePos->type & startSection){
            // a[] = {};
            if(nodePos->up->type & binaryOperatorType){
              ret += "{ ";
            }
            else{
              // Case: function(...) const {
              if( (((nodePos->sideDepth) != 0) &&
                   ((nodePos->up->down[nodePos->sideDepth - 1]->type & parentheses) ||
                    (nodePos->up->down[nodePos->sideDepth - 1]->value == "const")) )

                  || (nodePos->up->type & (occaKeywordType | flowControlType)))
                ret += " {\n" + tab + "  ";
              else
                ret += tab + "{\n";
            }

            tab += "  ";
          }
          else{
            tab = tab.substr(0, tab.size() - 2);

            // a[] = {};
            if(nodePos->up &&
               (nodePos->up->type & binaryOperatorType))
              ret += " }";
            else{
              ret += '}';

              //   }
              // }
              if((nodePos->up == NULL) ||
                 ((nodePos->up->right) &&
                  (nodePos->up->right->type == endBrace)))
                ret += "\n" + tab.substr(0, tab.size() - 2);
              else
                ret += "\n" + tab;
            }
          }
        }

        else if(nodePos->type == endParentheses){
          ret += ")";

          // if(...) statement
          if(autoMode)
            if((nodePos->up->type & flowControlType) &&
               (((nodePos->sideDepth) >= (nodePos->up->down.size() - 1)) ||
                !(nodePos->up->down[nodePos->sideDepth + 1]->type & brace))){

              ret += "\n" + tab + "  ";
            }
        }

        else if(nodePos->type & endStatement){
          ret += *nodePos;

          // for(){
          //   ...;
          // }
          if((nodePos->right == NULL) ||
             ((nodePos->right) &&
              (nodePos->right->type & brace))){

            ret += "\n" + tab.substr(0, tab.size() - 2);
          }
          //   blah;
          // }
          else if(!(nodePos->up)                    ||
                  !(nodePos->up->type & flowControlType) ||
                  !(nodePos->up->value == "for")){

            ret += "\n" + tab;
          }
          // Don't add newlines to for(A;B;C)
          else
            ret += " ";
        }

        else if(nodeHasDescriptor(nodePos)){
          ret += *nodePos;

          if(nodePos->right &&
             // [static ][const ][float ][variable]
             ((nodePos->right->type & (presetValue    |
                                       unknownVariable)) ||
              nodeHasDescriptor(nodePos->right))){

            ret += " ";
          }
        }

        else if(nodePos->type & flowControlType){
          ret += *nodePos;

          if(autoMode)
            if(nodePos->down.size() == 0)
              ret += '\n' + tab + "  ";
        }

        else if(nodePos->type & specialKeywordType){
          if(nodePos->value == "case")
            ret += "case";
          else if(nodePos->value == "default")
            ret += "default";
          else if(nodePos->value == "break")
            ret += "break";
          else if(nodePos->value == "continue")
            ret += "continue";
          else if(nodePos->value == "return"){
            ret += "return";

            if(nodePos->right || nodePos->down.size())
              ret += ' ';
          }
          else if(nodePos->value == "goto")
            ret += "goto ";
          else
            ret += *nodePos;
        }
        else if(nodePos->type & macroKeywordType){
          ret += *nodePos;

          ret += '\n' + tab;
        }
        else
          ret += *nodePos;

        const int downCount = (nodePos->down).size();

        for(int i = 0; i < downCount; ++i){
          strNode *downNode = nodePos->down[i];

          ret += prettyString(downNode, tab, autoMode);
        }

        nodePos = nodePos->right;
      }

      return ret;
#endif
    }

    statement::operator std::string(){
      const std::string tab = getTab();

      statementNode *statementPos = statementStart;

      // OCCA For's
      if(type == (occaStatementType | forStatementType)){
        std::string ret = tab + expRoot.toString() + "{\n";

        while(statementPos){
          ret += (std::string) *(statementPos->value);
          statementPos = statementPos->right;
        }

        ret += tab + "}\n";

        return ret;
      }

      else if(type & declareStatementType){
        return expRoot.toString(tab);
      }

      else if(type & (simpleStatementType | gotoStatementType)){
        return expRoot.toString(tab) + "\n";
      }

      else if(type & flowStatementType){
        std::string ret;

        if(type != doWhileStatementType){
          ret += expRoot.toString(tab);

          if(statementCount > 1)
            ret += "{";

          ret += '\n';
        }
        else{
          ret += tab;
          ret += "do {\n";
        }

        while(statementPos){
          ret += (std::string) *(statementPos->value);
          statementPos = statementPos->right;
        }

        if((statementCount > 1) ||
           (type == doWhileStatementType)){

            ret += tab + "}\n";
        }

        if(type == doWhileStatementType){
          ret += ' ';
          ret += expRoot.toString();
          ret += ";\n\n";
        }

        return ret;
      }

      else if(type & functionStatementType){
        if(type & functionDefinitionType){
          std::string ret = expRoot.toString(tab);

          ret += " {\n";

          while(statementPos){
            ret += (std::string) *(statementPos->value);
            statementPos = statementPos->right;
          }

          ret += tab + "}\n\n";

          return ret;
        }
        else if(type & functionPrototypeType)
          return expRoot.toString(tab);
      }
      else if(type & blockStatementType){
        std::string ret = "";

        if(0 <= depth)
          ret += tab + "{\n";

        while(statementPos){
          ret += (std::string) *(statementPos->value);
          statementPos = statementPos->right;
        }

        if(0 <= depth)
          ret += tab + "}\n";

        return ret;
      }
      else if(type & structStatementType){
        return expRoot.toString(tab) + "\n";
      }
      else if(type & macroStatementType){
        return expRoot.toString(tab);
      }

      return expRoot.toString(tab);
    }

    std::ostream& operator << (std::ostream &out, statement &s){
      out << (std::string) s;

      return out;
    }
    //==============================================
  };
};
