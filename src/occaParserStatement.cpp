#include "occaParserStatement.hpp"
#include "occaParser.hpp"

#include "occaTools.hpp"

namespace occa {
  namespace parserNS {
    //---[ Exp Node ]-------------------------------
    expNode::expNode() :
      sInfo(NULL),

      value(""),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL) {}

    expNode::expNode(statement &s) :
      sInfo(&s),

      value(""),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL) {}

    expNode::expNode(expNode &up_) :
      sInfo(up_.sInfo),

      value(""),
      info(expType::root),

      up(&up_),

      leafCount(0),
      leaves(NULL) {}

    bool expNode::operator == (expNode &e){
      return sameAs(e);
    }

    fnvOutput_t expNode::hash(){
      if(info & expType::hasInfo)
        return fnv(leaves[0]);

      fnvOutput_t fo = fnv(value);

      for(int i = 0; i < leafCount; ++i)
        fo.mergeWith(leaves[i]->hash());

      return fo;
    }

    bool expNode::sameAs(expNode &e, const bool nestedSearch){
      if(info != e.info)
        return false;

      if(info & expType::hasInfo)
        return (leaves[0] == e.leaves[0]);

      if(value != e.value)
        return false;

      if(!nestedSearch)
        return true;

      if(leafCount != e.leafCount)
        return false;

      if(hash() != e.hash())
        return false;

      bool *found = new bool[leafCount];

      for(int i = 0; i < leafCount; ++i)
        found[i] = false;

      // [-] Ugh, N^2
      for(int i = 0; i < leafCount; ++i){
        expNode &l1 = *(leaves[i]);
        bool iFound = false;

        for(int j = 0; j < leafCount; ++j){
          if(found[j])
            continue;

          expNode &l2 = *(e.leaves[j]);

          if(l1.sameAs(l2, !nestedSearch) &&
             l1.sameAs(l2,  nestedSearch)){

            iFound   = true;
            found[j] = true;
            break;
          }
        }

        if(!iFound)
          return false;
      }

      return true;
    }

    void expNode::loadFromNode(strNode *&nodePos, const bool parsingC){
      if(nodePos == NULL){
        sInfo->info = invalidStatementType;
        return;
      }

      strNode *nodeRoot = nodePos;

      sInfo->labelStatement(nodePos, this, parsingC);

      // Don't need to load stuff
      if(sInfo->info & (skipStatementType   |
                        macroStatementType  |
                        gotoStatementType   |
                        blockStatementType)            ||
         (sInfo->info == occaForType) ||
         (sInfo->info == elseStatementType)            ||
         (sInfo->info == doWhileStatementType)){

        return;
      }

      //---[ Special Type ]---
      if(nodeRoot->info & specialKeywordType){
        if((nodeRoot->value == "break")    ||
           (nodeRoot->value == "continue")){

          if((nodeRoot->value == "continue") &&
             (sInfo->distToOccaForLoop() <= sInfo->distToForLoop())){

            value = "occaContinue";
            info  = expType::transfer_;
          }
          else{
            value = nodeRoot->value;
            info  = expType::transfer_;
          }

          return;
        }

        // [-] Doesn't support GCC's twisted [Labels as Values]
        if(nodeRoot->value == "goto"){
          value = nodeRoot->right->value;
          info  = expType::goto_;
          return;
        }

        // Case where nodeRoot = [case, return]

        if((nodeRoot->value == "case") ||
           (nodeRoot->value == "default")){
          info = expType::checkSInfo;
        }
        else if(nodeRoot->value == "return"){
          info = expType::return_;
        }
      }
      //======================

      strNode *newNodeRoot = nodeRoot->cloneTo(nodePos);
      strNode *lastNewNode = lastNode(newNodeRoot);

      if(lastNewNode == NULL)
        newNodeRoot->print();

      if(parsingC)
        splitAndOrganizeNode(newNodeRoot);
      else
        splitAndOrganizeFortranNode(newNodeRoot);

      // std::cout << "[" << getBits(sInfo->info) << "] this = " << *this << '\n';
      // print();

      // Only the root needs to free
      if(up == NULL)
        occa::parserNS::free(newNodeRoot);
    }

    void expNode::splitAndOrganizeNode(strNode *nodeRoot){
      initLoadFromNode(nodeRoot);
      initOrganization();

      if(sInfo->info & declareStatementType)
        splitDeclareStatement();

      else if((sInfo->info & (ifStatementType    |
                              forStatementType   |
                              whileStatementType |
                              switchStatementType)) &&
              (sInfo->info != elseStatementType)){

        splitFlowStatement();
      }

      else if(sInfo->info & functionStatementType)
        splitFunctionStatement();

      else if(sInfo->info & structStatementType)
        splitStructStatement();

      else if(sInfo->info & caseStatementType)
        splitCaseStatement();

      else
        organize();
    }

    void expNode::splitAndOrganizeFortranNode(strNode *nodeRoot){
      initLoadFromFortranNode(nodeRoot);

      if(leaves[leafCount - 1]->value == "\\n")
        --leafCount;

      if(sInfo->info & declareStatementType)
        splitFortranDeclareStatement();

      if(sInfo->info & updateStatementType)
        splitFortranUpdateStatement();

      else if((sInfo->info & (ifStatementType  |
                              forStatementType |
                              whileStatementType)) &&
              (sInfo->info != elseStatementType)){

        splitFortranFlowStatement();
      }

      else if(sInfo->info & functionStatementType)
        splitFortranFunctionStatement();

      else if(sInfo->info & structStatementType)
        splitStructStatement();

      else if(sInfo->info & caseStatementType)
        splitCaseStatement(parsingFortran);

      else
        organize(parsingFortran);

      // std::cout << "this = " << *this << '\n';
    }

    void expNode::organize(const bool parsingC){
      if(leafCount == 0)
        return;

      if(parsingC)
        organizeLeaves();
      else
        organizeFortranLeaves();
    }

    void expNode::splitDeclareStatement(const int flags){
      info = expType::declaration;

      int varCount = 1 + typeInfo::delimiterCount(*this, ",");
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

        leaf[0].info |= expType::declaration;

        // Make sure the first one prints out the type
        if(i == 0){
          leaf[0].info |= expType::type;
          firstVar = &var;
        }

        removeNodes(leafPos, nextLeafPos - leafPos);

        int sExpStart = leafPos;
        int sExpEnd   = typeInfo::nextDelimiter(*this, leafPos, ",");

        leafPos = sExpEnd;

        // Don't put the [;]
        if((sExpEnd == leafCount) &&
           (leaves[sExpEnd - 1]->value == ";")){

          --sExpEnd;
        }

        if(sExpStart < sExpEnd){
          leaf.addNodes(expType::root, 1, sExpEnd - sExpStart);

          for(int j = sExpStart; j < sExpEnd; ++j)
            expNode::swap(*leaf.leaves[j - sExpStart + 1], *leaves[j]);

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

      int statementCount = 1 + typeInfo::delimiterCount(expDown, ";");

      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(expType::root, 0, statementCount);

      int leafPos = 0;

      for(int i = 0; i < statementCount; ++i){
        expNode &leaf = newExp[i];

        int nextLeafPos = typeInfo::nextDelimiter(expDown, leafPos, ";");

        if(leafPos < nextLeafPos){
          leaf.addNodes(expType::root, 0, (nextLeafPos - leafPos));

          for(int j = 0; j < leaf.leafCount; ++j){
            delete leaf.leaves[j];

            leaf.leaves[j]     = expDown.leaves[leafPos + j];
            leaf.leaves[j]->up = &leaf;
          }

          bool hasDeclare = ((sInfo->info & forStatementType) && (i == 0));

          if(hasDeclare &&
             ((leaf.leafCount == 0) ||
              !(leaf[0].info & (expType::qualifier |
                                expType::type      |
                                expType::typeInfo)))){

            hasDeclare = false;
          }

          if(!hasDeclare){
            leaf.organize();
          }
          else{
            leaf.splitDeclareStatement(expFlag::addVarToScope);

            expNode &flatRoot = *(makeFlatHandle());

            for(int j = 0; j < flatRoot.leafCount; ++j){
              expNode &n = flatRoot[j];

              // Variables that were just defined
              if(n.info & expType::unknown){
                varInfo *var = sInfo->hasVariableInLocalScope(n.value);

                if(var != NULL)
                  n.putVarInfo(*var);
              }
            }

            freeFlatHandle(flatRoot);
          }
        }

        leafPos = (nextLeafPos + 1);
      }

      expNode::swap(*this, newExp);
    }

    void expNode::splitFunctionStatement(const int flags){
      if(sInfo->info & functionDefinitionType)
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

      if(info == (expType::function | expType::declaration)){
        for(int i = 0; i < var.argumentCount; ++i)
          sInfo->addVariable(var.argumentVarInfos[i]);
      }

      removeNodes(1, leafPos);
    }

    void expNode::splitStructStatement(const int flags){
      info = expType::struct_;

      // Store type
      expNode newExp(*sInfo);
      newExp.info = info;

      typeInfo &type = newExp.addTypeInfoNode(0);
      type.loadFrom(*this, 0);

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

    void expNode::splitCaseStatement(const bool parsingC){
      // Fortran doesn't have [:] leaf at the end
      if(parsingC)
        --leafCount;

      // Remove [case] or [default]
      for(int i = 1; i < leafCount; ++i)
        leaves[i - 1] = leaves[i];

      --leafCount;
    }

    //  ---[ Fortran ]--------
    void expNode::splitFortranDeclareStatement(){
      info = expType::declaration;

      int varCount = 1;

      varInfo dummyVar;
      int varStart = dummyVar.loadTypeFromFortran(*this, 0);

      leafCount = typeInfo::nextDelimiter(*this, 0, "\\n");

      // [+] Needs to be updated on C++
      for(int i = varStart; i < leafCount; ++i){
        if(leaves[i]->value == ",")
          ++varCount;
      }

      int leafPos  = 0;

      varInfo *firstVar = NULL;

      // Store variables and stuff
      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(expType::root, 0, varCount);

      for(int i = 0; i < varCount; ++i){
        expNode &leaf = newExp[i];
        varInfo &var  = leaf.addVarInfoNode(0);

        int nextLeafPos = var.loadFromFortran(*this, leafPos, firstVar);

        if(var.stackPointerCount)
          var.stackPointersUsed = 1;

        leaf[0].info |= expType::declaration;

        // Make sure the first one prints out the type
        if(i == 0){
          leaf[0].info |= expType::type;
          firstVar = &var;
        }

        removeNodes(leafPos, nextLeafPos - leafPos);

        int sExpStart = leafPos;
        int sExpEnd   = typeInfo::nextDelimiter(*this, leafPos, ",");

        leafPos = sExpEnd;

        if(sExpStart < sExpEnd){
          leaf.addNodes(expType::root, 1, sExpEnd - sExpStart);

          for(int j = sExpStart; j < sExpEnd; ++j)
            expNode::swap(*leaf.leaves[j - sExpStart + 1], *leaves[j]);

          leaf.organizeFortranLeaves();
        }

        if(leafPos < leafCount)
          removeNode(leafPos);
      }

      expNode::swap(*this, newExp);

      //---[ Check DIMENSION ]----------
      if(firstVar->hasQualifier("DIMENSION")){
        for(int i = 1; i < varCount; ++i){
          varInfo &var = leaves[i]->getVarInfo(0);

          var.stackPointerCount = firstVar->stackPointerCount;
          var.stackPointersUsed = firstVar->stackPointersUsed;
          var.stackExpRoots     = firstVar->stackExpRoots;
        }

        firstVar->removeQualifier("DIMENSION");
      }

      //---[ Check INTENT ]-------------
      const bool hasIn    = firstVar->hasQualifier("INTENTIN");
      const bool hasOut   = firstVar->hasQualifier("INTENTOUT");
      const bool hasInOut = firstVar->hasQualifier("INTENTINOUT");

      if(hasIn || hasOut || hasInOut){
        for(int i = 0; i < varCount; ++i){
          varInfo &var = leaves[i]->getVarInfo(0);

          // Update here, also update in [Add variables to scope]
          //   for older Fortran codes without [INTENT]
          // Hide stack info in arguments
          var.stackPointersUsed = 0;

          // Make sure it registers as a pointer
          if((var.pointerCount      == 0) &&
             (var.stackPointerCount != 0)){

            var.pointerCount = 1;
            var.rightQualifiers.add("*", 0);
          }

          if(hasIn)
            var.removeQualifier("INTENTIN");
          else if(hasOut)
            var.removeQualifier("INTENTOUT");
          else if(hasInOut)
            var.removeQualifier("INTENTINOUT");

          varInfo *argVar = sInfo->hasVariableInScope(var.name);

          OCCA_CHECK(argVar != NULL,
                     "Error: variable [" << var << "] is not a function argument");

          *(argVar) = var;
        }

        sInfo->info = skipStatementType;
      }
      else{ // Add variables to scope
        for(int i = 0; i < varCount; ++i){

          varInfo &var = leaves[i]->getVarInfo(0);
          varInfo *pVar = sInfo->hasVariableInScope(var.name);

          // Check if it's a function argument
          if(pVar != NULL){
            statement *s = sInfo->getVarOriginStatement(*pVar);

            if(s &&
               (s->info & functionDefinitionType)){

              // Hide stack info in arguments
              var.stackPointersUsed = 0;

              // Make sure it registers as a pointer
              if((var.pointerCount      == 0) &&
                 (var.stackPointerCount != 0)){

                var.pointerCount = 1;
                var.rightQualifiers.add("*", 0);
              }

              *(pVar) = var;

              sInfo->info = skipStatementType;
            }
            // Will give error message
            else if(sInfo->up != NULL){
              sInfo->up->addVariable(&var, sInfo);
            }
          }
          else{
            if(sInfo->up != NULL){
              sInfo->up->addVariable(&var, sInfo);
            }
          }

        }
      }
    }

    void expNode::splitFortranUpdateStatement(){
      if(leafCount == 0)
        return;

      // Function call
      if(leaves[0]->value == "CALL"){
        // Only [CALL]
        if(leafCount == 1){
          sInfo->info = skipStatementType;
          return;
        }

        if(sInfo->hasVariableInScope(leaves[1]->value)){
          removeNode(0);

          leaves[0]->info = expType::function;

          mergeFunctionCalls();

          if(leaves[leafCount - 1]->value != ";")
            addNode(expType::operator_, ";");
        }
        else{
          OCCA_CHECK(false,
                     "Function [" << (leaves[0]->value) << "] is not defined in ["
                     << toString() << "]");
        }

        return;
      }

      organize(parsingFortran);

      varInfo *funcExp = sInfo->getFunctionVar();

      if((funcExp == NULL)            ||
         ((*this)[0].value    != "=") ||
         ((*this)[0][0].value != funcExp->name)){

        if(leaves[leafCount - 1]->value != ";")
          addNode(expType::operator_, ";");

        return;
      }

      expNode *retValueLeaf = &((*this)[0][1]);

      delete &((*this)[0][0]);
      delete &((*this)[0]);
      delete [] leaves;

      info = expType::return_;

      leaves    = new expNode*[2];
      leafCount = 2;

      leaves[0] = new expNode(*this);
      leaves[1] = retValueLeaf;

      (*this)[0].info  = expType::printValue;
      (*this)[0].value = "return";

      addNode(expType::operator_, ";");
    }

    void expNode::splitFortranFlowStatement(){
      info = expType::checkSInfo;

      if(leafCount == 0)
        return;

      if(sInfo->info & forStatementType){
        splitFortranForStatement();
      }
      // [IF/ELSE IF/DO WHILE]( EXPR )
      else if((sInfo->info == ifStatementType)     ||
              (sInfo->info == elseIfStatementType) ||
              (sInfo->info == whileStatementType)){

        OCCA_CHECK(leafCount != 0,
                   "No expression in if-statement: " << *this << '\n');

        leaves[0]       = leaves[1];
        leaves[0]->info = expType::root;
        leaves[0]->organize();

        leafCount = 1;
      }
      // [ELSE]
      else if(sInfo->info & elseStatementType){
        if(leafCount)
          free();
      }
    }

    void expNode::splitFortranForStatement(){
      // [DO] iter=start,end[,stride][,loop]
      // Infinite [DO]
      if(leafCount == 1){
        leaves[0]->value = "true";
        leaves[0]->info  = presetValue;
        leafCount = 1;

        sInfo->info = whileStatementType;

        return;
      }

      leafCount = typeInfo::nextDelimiter(*this, 0, "\\n");

      int statementCount = 1 + typeInfo::delimiterCount(*this, ",");

      OCCA_CHECK((2 <= statementCount) && (statementCount <= 4),
                 "Error: Wrong [DO] format [" << *this << "]");

      int pos[5];

      // Skip [DO], [iter], and [=]
      pos[0] = 3;

      // Find [,] positions
      for(int i = 0; i < statementCount; ++i){
        pos[i + 1] = typeInfo::nextDelimiter(*this, pos[i], ",") + 1;

        OCCA_CHECK(pos[i] != (pos[i + 1] + 1),
                   "Error: No expression given in [" << *this << "]");
      }

      // Check if last expressiong is an OCCA tag
      std::string &lastLeafValue = leaves[pos[statementCount - 1]]->value;

      const bool hasOccaTag = isAnOccaTag(lastLeafValue);

      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(expType::root, 0, 3 + hasOccaTag);

      if(hasOccaTag){
        expNode &leaf = newExp[3];
        leaf.addNode(expType::presetValue, 0);
        leaf[0].value = lastLeafValue;

        // Get rid of the tag
        --statementCount;
      }

      // Get iter var name
      const std::string &iter = leaves[1]->value;
      varInfo *var = sInfo->hasVariableInScope(iter);

      OCCA_CHECK(var != NULL,
                 "Error: Iterator [" << iter
                 << "] is not defined before [" << *this << "]");

      // Fortran iterations are not modified
      std::vector<std::string> doNames;

      doNames.push_back("doStart");
      doNames.push_back("doEnd");
      doNames.push_back("doStride");
      doNames.push_back("doStrideSign");

      std::string &doStart      = doNames[0];
      std::string &doEnd        = doNames[1];
      std::string &doStride     = doNames[2];
      std::string &doStrideSign = doNames[3];

      sInfo->createUniqueVariables(doNames);

      const std::string exp0 = toString(pos[0], (pos[1] - pos[0] - 1));
      const std::string exp1 = toString(pos[1], (pos[2] - pos[1] - 1));

      const std::string decl0 = "const int " + doStart + " = " + exp0;
      const std::string decl1 = "const int " + doEnd   + " = " + exp1;

      OCCA_CHECK(exp0.size() != 0,
                 "Error, missing 1st statement in the [DO]: " << toString() << '\n');
      OCCA_CHECK(exp1.size() != 0,
                 "Error, missing 2nd statement in the [DO]: " << toString() << '\n');

      sInfo->up->addStatementFromSource(decl0);
      sInfo->up->addStatementFromSource(decl1);

      if(statementCount == 3){
        const std::string exp2  = toString(pos[2], (pos[3] - pos[2] - 1));
        const std::string decl2 = "const int " + doStride + " = " + exp2;

        OCCA_CHECK(exp2.size() != 0,
                   "Error, missing 3rd statement in the [DO]: " << toString() << '\n');

        sInfo->up->addStatementFromSource(decl2);

        const std::string decl3 = "const int " + doStrideSign + " = (1 - (2*(" + doStride + " < 0)))";

        sInfo->up->addStatementFromSource(decl3);
      }

      newExp.leaves[0] = sInfo->createExpNodeFrom(iter + " = " + doStart);

      if(statementCount == 3){
        newExp.leaves[1] = sInfo->createExpNodeFrom("0 <= (" + doStrideSign + "* (" + doEnd + " - " + iter + "))");
        newExp.leaves[2] = sInfo->createExpNodeFrom(iter + " += " + doStride);
      }
      else{
        newExp.leaves[1] = sInfo->createExpNodeFrom(iter + " <= " + doEnd);
        newExp.leaves[2] = sInfo->createExpNodeFrom("++" + iter);
      }

      newExp[0].labelUsedVariables();
      newExp[1].labelUsedVariables();
      newExp[2].labelUsedVariables();

      varInfo &vDoStart = *(sInfo->hasVariableInScope(doStart));
      varInfo &vDoEnd   = *(sInfo->hasVariableInScope(doEnd));

      sInfo->addVariableToUsedMap(vDoStart);
      sInfo->addVariableToUsedMap(vDoEnd);

      if(statementCount == 3){
        varInfo &vDoStride     = *(sInfo->hasVariableInScope(doStride));
        varInfo &vDoStrideSign = *(sInfo->hasVariableInScope(doStrideSign));

        sInfo->addVariableToUsedMap(vDoStride);
        sInfo->addVariableToUsedMap(vDoStrideSign);
      }

      expNode::swap(*this, newExp);
    }

    void expNode::splitFortranFunctionStatement(){
      info = (expType::function | expType::declaration);

      if(leafCount == 0)
        return;

      varInfo &var = addVarInfoNode(0);
      int leafPos  = var.loadFromFortran(*this, 1);

      if((sInfo->up != NULL)              &&
         (sInfo->up->scopeVarMap.find(var.name) ==
          sInfo->up->scopeVarMap.end())){

        sInfo->up->addVariable(&var);

        // Add initial arguments (they get updated later)
        for(int i = 0; i < var.argumentCount; ++i)
          sInfo->addVariable( &(var.getArgument(i)) );
      }

      removeNodes(1, leafPos);
    }
    //  ======================

    void expNode::translateOccaKeyword(strNode *nodePos, const bool parsingC){
      if(nodePos->info & occaKeywordType){

        if(((parsingC)  &&
            (nodePos->value == "directLoad")) ||
           ((!parsingC) &&
            upStringCheck(nodePos->value, "DIRECTLOAD"))){

          nodePos->value = "occaDirectLoad";
        }

      }
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
        if(nodePos->info & occaKeywordType)
          translateOccaKeyword(nodePos, true);

        expNode *&leaf = leaves[leafPos++];

        leaf        = new expNode(*this);
        leaf->value = nodePos->value;

        if(nodePos->info & unknownVariable){
          varInfo *nodeVar = sInfo->hasVariableInScope(nodePos->value);

          if(nodeVar){
            if( !(nodeVar->info & varType::functionType) )
              leaf->putVarInfo(*nodeVar);
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

        else if(nodePos->info & presetValue){
          leaf->info = expType::presetValue;
        }

        else if(nodePos->info & descriptorType){
          if(nodePos->info == keywordType["long"]){
            if((nodePos->right) &&
               (sInfo->hasTypeInScope(nodePos->right->value))){

              leaf->info = expType::qualifier;
            }
            else
              leaf->info = expType::type;
          }
          else if(nodePos->info & (qualifierType | structType))
            leaf->info = expType::qualifier;
          else
            leaf->info = expType::type;

          // For [*] and [&]
          if(nodePos->info & operatorType)
            leaf->info |= expType::operator_;
        }

        else if(nodePos->info & structType){
          leaf->info = expType::qualifier;
        }

        else if(nodePos->info & operatorType){
          leaf->info = expType::operator_;
        }

        else if(nodePos->info & startSection){
          leaf->info  = expType::C;

          if(nodePos->down)
            leaf->initLoadFromNode(nodePos->down);
        }

        else
          leaf->info = expType::printValue;

        if(nodePos->info == 0){
          delete leaf;
          --leafPos;
        }

        nodePos = nodePos->right;
      }
    }

    void expNode::initLoadFromFortranNode(strNode *nodeRoot){
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

        if(nodePos->info & unknownVariable){
          varInfo *nodeVar = sInfo->hasVariableInScope(nodePos->value);

          if(nodeVar)
            leaf->putVarInfo(*nodeVar);
          else
            leaf->info = expType::unknown;
        }

        else if(nodePos->info & presetValue){
          leaf->info = expType::presetValue;
        }

        else if(nodePos->info & descriptorType){
          if(nodePos->info & qualifierType)
            leaf->info = expType::qualifier;
          else
            leaf->info  = expType::type;
        }

        else if(nodePos->info & operatorType){
          leaf->info = expType::operator_;
        }

        else if(nodePos->info & startSection){
          leaf->info  = expType::C;

          if(nodePos->down)
            leaf->initLoadFromNode(nodePos->down);
        }

        else
          leaf->info = expType::printValue;

        if(nodePos->info == 0){
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
           !(leaves[i]->info & expType::hasInfo)){

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

    void expNode::organizeLeaves(const bool inRoot){
      if(info & expType::hasInfo)
        return;

      // Organize leaves bottom -> up
      for(int i = 0; i < leafCount; ++i){
        if((leaves[i]->leafCount) &&
           !(leaves[i]->info & expType::hasInfo)){

          leaves[i]->organizeLeaves(false);
        }
      }

      // Add used vars to varUsedMap
      if(inRoot)
        labelUsedVariables();

      //---[ Level 1 ]------
      // <const int,float>
      mergeTypes();

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

    void expNode::organizeFortranLeaves(){
      if(info & expType::hasInfo)
        return;

      // Organize leaves bottom -> up
      for(int i = 0; i < leafCount; ++i){
        if((leaves[i]->leafCount) &&
           !(leaves[i]->info & expType::hasInfo)){

          leaves[i]->organizeFortranLeaves();
        }
      }

      mergeFortranArrays();

      for(int i = 0; i < 12; ++i)
        organizeLeaves(i);

      translateFortranMemberCalls();
      translateFortranPow();
    }

    void expNode::organizeLeaves(const int level){
      bool leftToRight = opLevelL2R[level];

      int leafPos  = (leftToRight ? 0 : (leafCount - 1));
      const int ls = (leftToRight ? 1 : -1);

      while(true){
        if(( (leftToRight) && (leafCount <= leafPos)) ||
           ((!leftToRight) && (leafPos < 0)))
          break;

        if((leaves[leafPos]->leafCount)                  ||
           (leaves[leafPos]->info &  expType::hasInfo)   ||
           (leaves[leafPos]->info == expType::qualifier)){

          leafPos += ls;
          continue;
        }

        std::string &lStr = leaves[leafPos]->value;
        opLevelMapIterator it = opLevelMap[level].find(lStr);

        if(it == opLevelMap[level].end()){
          leafPos += ls;
          continue;
        }

        const int levelType = it->second;

        if(levelType & unitaryOperatorType){
          bool updateNow = true;

          const int targetOff = ((levelType & lUnitaryOperatorType) ? 1 : -1);
          const int target    = leafPos + targetOff;

          if((target < 0) || (leafCount <= target)){
            updateNow = false;
          }
          else{
            if(leaves[target]->info & expType::operator_)
              updateNow = false;
            // Cases: & * + -
            else if(keywordType[lStr] & binaryOperatorType){
              const int invTarget = leafPos + ((targetOff == 1) ?
                                               -1 : 1);

              updateNow = false;

              if((invTarget < 0) || (leafCount <= invTarget) ||
                 (leaves[invTarget]->info & expType::operator_)){

                updateNow = true;
              }
            }
          }

          if(!updateNow){
            leafPos += ls;
          }
          else{
            if(levelType & lUnitaryOperatorType)
              leafPos = mergeLeftUnary(leafPos, leftToRight);
            else
              leafPos = mergeRightUnary(leafPos, leftToRight);
          }
        }
        else if(levelType & binaryOperatorType)
          leafPos = mergeBinary(leafPos, leftToRight);
        else if(levelType & ternaryOperatorType)
          leafPos = mergeTernary(leafPos, leftToRight);
        else
          leafPos += ls;
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
        if(leaves[leafPos]->info != (expType::operator_ |
                                     expType::qualifier)){

          ++leafPos;
          continue;
        }

        expNode &leaf = *(leaves[leafPos]);

        if(leafPos == 0){
          leaf.info = expType::operator_;
          ++leafPos;
          continue;
        }

        expNode &lLeaf = *(leaves[leafPos - 1]);

        if(lLeaf.info & (expType::qualifier |
                         expType::type)){

          leaf.info = expType::qualifier;
        }

        else if(lLeaf.info & expType::unknown){
          if(!sInfo->hasTypeInScope(lLeaf.value))
            leaf.info = expType::operator_;
          else
            leaf.info = expType::qualifier;
        }

        else if((lLeaf.value == ",") &&
                (up == NULL)         &&
                (sInfo->info == declareStatementType)){

          leaf.info = expType::qualifier;
        }

        else{
          leaf.info = expType::operator_;
        }

        ++leafPos;
      }
    }

    // [(class)]
    void expNode::labelCasts(){
      // Don't mistake:
      //   int main(int) -> int main[(int)]
      if(sInfo->info & functionStatementType)
        return;

      int leafPos = 0;

      while(leafPos < leafCount){
        expNode &leaf = *(leaves[leafPos]);

        if((leaf.value == "(")                      &&
           (leaf.leafCount)                         &&
           ((leaf[0].info & (expType::type      |
                             expType::qualifier |
                             expType::typeInfo))     ||
            sInfo->hasTypeInScope(leaves[leafPos]->value))){

          bool isCast = true;

          for(int i = 1; i < leaf.leafCount; ++i){
            if(!(leaf[i].info & (expType::type      |
                                 expType::qualifier |
                                 expType::typeInfo))     &&
               !sInfo->hasTypeInScope(leaves[leafPos]->value)){

              isCast = false;
              break;
            }
          }

          if(isCast)
            leaf.info = expType::cast_;
        }

        ++leafPos;
      }
    }

    // Add used vars to varUsedMap
    void expNode::labelUsedVariables(){
      for(int i = 0; i < leafCount; ++i){
        expNode &n = *(leaves[i]);

        if(n.hasVariable()){
          std::string varName = n.getMyVariableName();
          varInfo &var        = *(sInfo->hasVariableInScope(varName));

          if(((i + 1) < leafCount) &&
             isAnUpdateOperator(leaves[i + 1]->value)){

            sInfo->addVariableToUpdateMap(var);
          }
          else{
            sInfo->addVariableToUsedMap(var);
          }
        }
      }
    }

    // <const int,float>
    void expNode::mergeTypes(){
      int leafPos = 0;

      while(leafPos < leafCount){
        if(sInfo->hasTypeInScope(leaves[leafPos]->value) ||
           (leaves[leafPos]->info == expType::qualifier)){

          varInfo &var = addVarInfoNode(leafPos);

          leaves[leafPos++]->info |= expType::type;

          const int nextLeafPos = var.loadFrom(*this, leafPos);

          removeNodes(leafPos, nextLeafPos - leafPos);
        }

        ++leafPos;
      }
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
            newLeaf->addNode(expType::qualifier);
            newLeaf->leaves[newLeaf->leafCount - 1] = sNewLeaf;
          }
          else{
            newLeaf->leaves[0] = leaves[leafPos - 1];
            newLeaf->leaves[1] = sNewLeaf;

            newLeaf->leaves[0]->up = newLeaf;

            leaves[leafPos - 1] = newLeaf;
          }

          for(int i = 0; i < brackets; ++i){
            sNewLeaf->leaves[i]     = leaves[leafPos + i];
            leaves[leafPos + i]->up = sNewLeaf;
          }

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
      int leafPos = leafCount - 2;

      while(0 <= leafPos){
        if( !(leaves[leafPos]->info & expType::cast_) ){
          --leafPos;
          continue;
        }

        expNode *leaf  = leaves[leafPos];
        expNode *sLeaf = leaves[leafPos + 1];

        for(int i = (leafPos + 2); i < leafCount; ++i)
          leaves[i - 1] = leaves[i];

        --leafCount;

        leaf->addNode(*sLeaf);

        sLeaf->up = leaf;

        --leafPos;
      }
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
    int expNode::mergeLeftUnary(const int leafPos, const bool leftToRight){
      const int retPos = (leftToRight ? (leafPos + 1) : (leafPos - 1));

      if(leafCount <= (leafPos + 1))
        return retPos;

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

      return retPos;
    }

    // i[++]
    int expNode::mergeRightUnary(const int leafPos, const bool leftToRight){
      const int retPos = (leftToRight ? (leafPos + 1) : (leafPos - 1));

      if(0 == leafPos)
        return retPos;

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

      return retPos;
    }

    // a [+] b
    int expNode::mergeBinary(const int leafPos, const bool leftToRight){
      const int retPos = (leftToRight ? (leafPos + 1) : (leafPos - 1));

      if((0 == leafPos) || (leafCount <= (leafPos + 1)))
        return retPos;

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

      return (leftToRight ? leafPos : leafPos - 2);
    }

    // a [?] b : c
    int expNode::mergeTernary(const int leafPos, const bool leftToRight){
      const int retPos = (leftToRight ? (leafPos + 1) : (leafPos - 1));

      if((0 == leafPos) || (leafCount <= (leafPos + 3)))
        return retPos;

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

      return (leftToRight ? leafPos : leafPos - 2);
    }

    //---[ Custom Type Info ]---------
    bool expNode::qualifierEndsWithStar(){
      if( !(info & expType::qualifier) )
        return false;

      if(leafCount)
        return leaves[leafCount - 1]->qualifierEndsWithStar();
      else
        return (value == "*");
    }

    bool expNode::typeEndsWithStar(){
      if( !(info & expType::type) ||
          (leafCount == 0) )
        return false;

      if(leaves[leafCount - 1]->info & expType::qualifier)
        return leaves[leafCount - 1]->qualifierEndsWithStar();

      return false;
    }

    bool expNode::hasAnArrayQualifier(const int pos){
      if( !(info & expType::qualifier) ||
          (leafCount <= pos) )
        return false;

      return ((leaves[pos]->value == "*") ||
              (leaves[pos]->value == "&") ||
              (leaves[pos]->value == "["));
    }

    void expNode::mergeFortranArrays(){
      int leafPos = 0;

      while(leafPos < leafCount){
        expNode &leaf = *(leaves[leafPos]);

        if((leaf.value == "(")                            && // Is ()
           (leaf.leafCount)                               && //   and has stuff
           (0 < leafPos)                                  && //   and follows
           (leaves[leafPos - 1]->info & (expType::varInfo  | //   something
                                         expType::unknown))){

          expNode *pLeaf = &(leaf[0]);
          int entries = 1;

          while(pLeaf &&
                (pLeaf->value == ",") &&
                (pLeaf->leafCount)){

            ++entries;
            pLeaf = pLeaf->leaves[0];
          }

          if(entries == 1) {
            leaf.value = "[";

            if(leaf.leafCount){
              expNode &sLeaf = leaf[0];

              leaf.leafCount = 0;

              leaf.addNode(expType::LR, "-");

              leaf[0].addNode( sLeaf );
              leaf[0].addNode(expType::presetValue, "1");
            }
          }
          else {
            expNode &varLeaf = *(leaves[leafPos - 1]);
            varInfo *pVar = sInfo->hasVariableInScope(varLeaf.value);

            const bool mergeEntries = ((pVar != NULL) &&
                                       (pVar->stackPointersUsed <= 1));

            expNode newExp(*sInfo);
            newExp.info = info;
            newExp.addNodes(expType::root, 0, entries);

            pLeaf = &(leaf[0]);

            while(pLeaf &&
                  (pLeaf->value == ",") &&
                  (pLeaf->leafCount)){

              delete (newExp.leaves[--entries]);

              newExp.leaves[entries] = pLeaf->leaves[1];

              pLeaf = pLeaf->leaves[0];
            }

            if(entries)
              newExp.leaves[--entries] = pLeaf;

            entries = newExp.leafCount;

            if(!mergeEntries){
              addNodes(expType::C, leafPos, (entries - 1));

              for(int i = 0; i < entries; ++i){
                expNode &sLeaf = *(leaves[leafPos + i]);

                sLeaf.value     = "[";
                sLeaf.leafCount = 0;

                sLeaf.addNode(expType::LR, "-");

                sLeaf[0].addNode( newExp[entries - i - 1] );
                sLeaf[0].addNode(expType::presetValue, "1");
              }
            }
            else{
              expNode *cpLeaf = &leaf;
              varInfo &var    = *pVar;

              if(entries != var.stackPointerCount){
                // Revert [var] back to original dimensions
                var.stackPointersUsed = var.stackPointerCount;

                // [+] Print [var] Fortran-style
                OCCA_CHECK(false,
                           "Incorrect dimensions on variable ["
                           << var << "], in statement ["
                           << *(leaf.up) << "]");
              }

              leaf.value = "[";

              if(leaf.leafCount){
                delete [] leaf.leaves;
                leaf.leafCount = 0;
              }

              for(int i = 0; i < (entries - 1); ++i){
                expNode &cLeaf = *cpLeaf;

                cLeaf.addNode(expType::LR, "+");

                expNode &plusLeaf = cLeaf[0];

                plusLeaf.addNodes(expType::C, 0, 2);

                plusLeaf[0].value = "(";
                plusLeaf[1].value = "(";

                plusLeaf[0].addNode(expType::C, "(");

                expNode &offLeaf = plusLeaf[0][0];

                offLeaf.addNode(expType::LR, "-");
                offLeaf[0].addNode( newExp[i] );
                offLeaf[0].addNode(expType::presetValue, "1");

                plusLeaf[1].addNode(expType::LR, "*");

                expNode &nextLeaf = plusLeaf[1][0];

                nextLeaf.addNode(expType::C , "(");
                nextLeaf.addNode(expType::C , "(");

                nextLeaf[0].addNode(var.stackSizeExpNode(entries - i - 1));

                cpLeaf = &(nextLeaf[1]);
              }

              cpLeaf->addNode(expType::LR, "-");

              expNode &lcpLeaf = (*cpLeaf)[-1];

              lcpLeaf.addNode( newExp[entries - 1] );
              lcpLeaf.addNode(expType::presetValue, "1");
            }

            delete [] newExp.leaves;

            leafPos += (entries - 1);
          }
        }

        ++leafPos;
      }
    }

    void expNode::translateFortranMemberCalls(){
      expNode &flatRoot = *(makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if((n.info == expType::LR) &&
           (n.value == "%")){

          n.value = ".";
        }
      }

      freeFlatHandle(flatRoot);
    }

    void expNode::translateFortranPow(){
      expNode &flatRoot = *(makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if((n.info == expType::LR) &&
           (n.value == "**")){

          expNode &x = n[0];
          expNode &y = n[1];

          n.value     = "occaPow";
          n.info      = expType::function;
          n.leafCount = 0;

          n.addNode(expType::C, "(");

          expNode &sLeaf1 = n[0];
          sLeaf1.addNode(expType::LR, ",");

          expNode &sLeaf2 = sLeaf1[0];

          sLeaf2.addNode(x);
          sLeaf2.addNode(y);
        }
      }

      freeFlatHandle(flatRoot);
    }
    //================================

    void expNode::swap(expNode &a, expNode &b){
      swapValues(a.sInfo, b.sInfo);

      swapValues(a.value, b.value);
      swapValues(a.info , b.info);

      swapValues(a.up, b.up);

      swapValues(a.leafCount, b.leafCount);
      swapValues(a.leaves   , b.leaves);

      if( !(a.info & expType::hasInfo) ){
        for(int i = 0; i < a.leafCount; ++i)
          a.leaves[i]->up = &a;
      }

      if( !(b.info & expType::hasInfo) ){
        for(int i = 0; i < b.leafCount; ++i)
          b.leaves[i]->up = &b;
      }
    }

    expNode* expNode::clone(){
      expNode &newRoot = *(new expNode(*sInfo));

      cloneTo(newRoot);

      return &newRoot;
    }

    expNode* expNode::clone(statement &s){
      expNode &newRoot = *(new expNode(s));

      cloneTo(newRoot);

      return &newRoot;
    }

    expNode* expNode::clone(expNode *original){
      expNode &newLeaf = *(new expNode(*this));

      original->cloneTo(newLeaf);

      return &newLeaf;
    }

    void expNode::cloneTo(expNode &newExp){
      statement *sUp = ((newExp.sInfo != NULL) ?
                        newExp.sInfo->up       :
                        NULL);

      const bool sChanged = ((newExp.sInfo != NULL) &&
                             (newExp.sInfo != sInfo));

      newExp.info = info;

      const bool isVarInfo  = (info & expType::varInfo);
      const bool isTypeInfo = (info & expType::typeInfo);
      const bool isFuncInfo = ((info == (expType::function |
                                         expType::declaration)) ||
                               (info == (expType::function |
                                         expType::prototype)));

      const bool inForStatement = ((newExp.sInfo != NULL) &&
                                   (newExp.sInfo->info & forStatementType));

      if(isVarInfo | isTypeInfo | isFuncInfo){
        if(isVarInfo){
          // Var is created if it also has [expType::type]
          if(info & expType::declaration){
            varInfo &var = newExp.addVarInfoNode();
            var = getVarInfo().clone();

            // addVarInfoNode() sets info
            newExp.info = info;

            if(sChanged){
              if(inForStatement){
                newExp.sInfo->addVariable(&var);
              }
              else if((sUp != NULL) &&
                      !(sUp->hasVariableInLocalScope(var.name))){

                sUp->addVariable(&var, newExp.sInfo);
              }
            }
          }
          else{ // (info == expType::varInfo)
            varInfo &var = getVarInfo();

            newExp.putVarInfo(var);

            newExp.info = info;

            if((sInfo        != NULL) &&
               (newExp.sInfo != NULL)) {

              if((newExp.up != NULL) &&
                 (!isAnUpdateOperator(newExp.up->value))){

                sInfo->addVariableToUpdateMap(var, newExp.sInfo);
              }
              else{
                sInfo->addVariableToUsedMap(var, newExp.sInfo);
              }
            }
          }
        }
        else if(isTypeInfo){
          typeInfo &type = newExp.addTypeInfoNode();
          type = getTypeInfo().clone();

          if(sChanged      &&
             (sUp != NULL) &&
             !(sUp->hasVariableInLocalScope(type.name))){

            sUp->addType(type);
          }
        }
        else if(isFuncInfo){
          newExp.addVarInfoNode(0);
          newExp.setVarInfo(0, leaves[0]->getVarInfo());

          // Get function variable
          varInfo &var = leaves[0]->getVarInfo();

          // Make sure we haven't initialized it
          //   from the original or an extern
          if(sChanged      &&
             (sUp != NULL) &&
             !(sUp->hasVariableInLocalScope(var.name)) ){

            sUp->addVariable(&var);
          }

          for(int i = 0; i < var.argumentCount; ++i){
            varInfo &argVar = *(new varInfo());
            argVar = var.getArgument(i).clone();

            newExp.sInfo->addVariable(&argVar);
            var.setArgument(i, argVar);
          }
        }
      }
      else {
        newExp.value     = value;
        newExp.leafCount = leafCount;

        if(leafCount){
          newExp.leaves = new expNode*[leafCount];

          for(int i = 0; i < leafCount; ++i){
            newExp.leaves[i] = new expNode(newExp);
            leaves[i]->cloneTo(newExp[i]);
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
      expNode *up_ = up;
      int depth_   = 0;

      while(up_){
        ++depth_;
        up_ = up_->up;
      }

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
      if(info & expType::hasInfo)
        return 0;

      int ret = leafCount;

      for(int i = 0; i < leafCount; ++i){
        if(leaves[i]->leafCount)
          ret += leaves[i]->nestedLeafCount();
      }

      return ret;
    }

    expNode* expNode::makeFlatHandle(){
      expNode *flatNode;

      if(sInfo != NULL)
        flatNode = new expNode(*sInfo);
      else
        flatNode = new expNode(*this);

      const bool addMe = (info != 0);

      if((leafCount == 0) && !addMe)
        return flatNode;

      flatNode->info   = expType::printLeaves;
      flatNode->leaves = new expNode*[addMe + nestedLeafCount()];

      int offset = 0;
      makeFlatHandle(offset, flatNode->leaves);

      if(addMe)
        flatNode->setLeaf(*this, offset++);

      flatNode->leafCount = offset;

      return flatNode;
    }

    void expNode::makeFlatHandle(int &offset,
                                 expNode **flatLeaves){
      if(info & expType::hasInfo)
        return;

      for(int i = 0; i < leafCount; ++i){
        switch(leaves[i]->info){
        case (expType::L):{
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i];
          flatLeaves[offset++] = leaves[i]->leaves[0];

          break;
        }

        case (expType::R):{
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i]->leaves[0];
          flatLeaves[offset++] = leaves[i];

          break;
        }

        case (expType::L | expType::R):{
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          leaves[i]->leaves[1]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i]->leaves[0];
          flatLeaves[offset++] = leaves[i];
          flatLeaves[offset++] = leaves[i]->leaves[1];

          break;
        }

        case (expType::L | expType::C | expType::R):{
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          leaves[i]->leaves[1]->makeFlatHandle(offset, flatLeaves);
          leaves[i]->leaves[2]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i]->leaves[0];
          flatLeaves[offset++] = leaves[i]->leaves[1];
          flatLeaves[offset++] = leaves[i]->leaves[2];

          break;
        }
        default:
          leaves[i]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i];

          break;
        }
      }
    }

    void expNode::freeFlatHandle(expNode &flatRoot){
      if(flatRoot.leafCount)
        delete [] flatRoot.leaves;

      delete &flatRoot;
    }

    expNode* expNode::makeCsvFlatHandle(){
      expNode *flatNode;

      if(sInfo != NULL)
        flatNode = new expNode(*sInfo);
      else
        flatNode = new expNode(*this);

      if((leafCount == 0) && (info == expType::root))
        return flatNode;

      int csvCount = 1;

      for(int pass = 0; pass < 2; ++pass){
        expNode *cNode = ((info == expType::root) ? leaves[0] : this);

        if(pass == 1){
          flatNode->info      = expType::printLeaves;
          flatNode->leaves    = new expNode*[csvCount];
          flatNode->leafCount = csvCount;
        }

        while(cNode                         &&
              (cNode->info  == expType::LR) &&
              (cNode->value == ",")){

          if(pass == 0){
            ++csvCount;
          }
          else {
            flatNode->leaves[--csvCount] = cNode->leaves[1];

            if(csvCount == 1)
              flatNode->leaves[--csvCount] = cNode->leaves[0];
          }

          cNode = cNode->leaves[0];
        }

        if((pass == 1) && csvCount)
          flatNode->leaves[0] = cNode;
      }

      return flatNode;
    }

    void expNode::addNodes(const int info_,
                           const int pos_,
                           const int count){

      const int pos = ((0 <= pos_) ? pos_ : leafCount);

      reserveAndShift(pos, count);

      for(int i = pos; i < (pos + count); ++i){
        leaves[i] = new expNode(*this);

        leaves[i]->info      = info_;
        leaves[i]->leafCount = 0;
        leaves[i]->leaves    = NULL;
      }
    }

    void expNode::addNode(const int info_,
                          const int pos){
      addNodes(info_, pos, 1);
    }

    void expNode::addNode(const int info_,
                          const std::string &value_,
                          const int pos){

      addNodes(info_, pos, 1);

      if(0 <= pos)
        leaves[pos]->value = value_;
      else
        leaves[leafCount - 1]->value = value_;
    }

    void expNode::addNode(expNode &node_,
                          const int pos_){

      const int pos = ((0 <= pos_) ? pos_ : leafCount);

      reserveAndShift(pos, 1);

      leaves[pos] = &node_;

      node_.up = this;
    }

    void expNode::reserve(const int count){
      reserveAndShift(0, count);
    }

    void expNode::reserveAndShift(const int pos,
                                  const int count){

      expNode **newLeaves = new expNode*[leafCount + count];

      //---[ Add Leaves ]-----
      for(int i = 0; i < pos; ++i)
        newLeaves[i] = leaves[i];

      for(int i = pos; i < leafCount; ++i)
        newLeaves[i + count] = leaves[i];
      //======================

      if(leafCount)
        delete [] leaves;

      leaves = newLeaves;

      leafCount += count;
    }

    void expNode::setLeaf(expNode &leaf, const int pos){
      leaves[pos] = &leaf;
      leaf.up     = this;
    }

    varInfo& expNode::addVarInfoNode(){
      addNode(0);

      varInfo **varLeaves = (varInfo**) leaves;
      varInfo *&varLeaf   = varLeaves[0];

      varLeaf = new varInfo();
      return *varLeaf;
    }

    varInfo& expNode::addVarInfoNode(const int pos_){
      const int pos = ((0 <= pos_) ? pos_ : leafCount);

      addNode(expType::varInfo, pos);

      return leaves[pos]->addVarInfoNode();
    }

    void expNode::putVarInfo(varInfo &var){
      addNode(0);
      leaves[0] = (expNode*) &var;

      info = expType::varInfo;
    }

    void expNode::putVarInfo(const int pos, varInfo &var){
      addNode(expType::varInfo, pos);
      leaves[pos]->putVarInfo(var);
    }

    typeInfo& expNode::addTypeInfoNode(){
      addNode(0);

      typeInfo **typeLeaves = (typeInfo**) leaves;
      typeInfo *&typeLeaf   = typeLeaves[0];

      typeLeaf = new typeInfo();
      return *typeLeaf;
    }

    typeInfo& expNode::addTypeInfoNode(const int pos){
      addNode(expType::typeInfo, pos);
      return leaves[pos]->addTypeInfoNode();
    }

    bool expNode::hasVariable(){
      if(info & (expType::variable |
                 expType::varInfo  |
                 expType::function)){

        if( (info & expType::varInfo) ||
            (value.size() != 0) ){

          return true;
        }
      }

      return false;
    }

    varInfo& expNode::getVarInfo(){
      if(info & expType::variable)
        return leaves[0]->getVarInfo();

      return *((varInfo*) leaves[0]);
    }

    varInfo& expNode::getVarInfo(const int pos_){
      const int pos = ((0 <= pos_) ? pos_ : leafCount);

      varInfo **varLeaves = (varInfo**) leaves[pos]->leaves;
      varInfo *&varLeaf   = varLeaves[0];

      return *varLeaf;
    }

    void expNode::setVarInfo(varInfo &var){
      leaves[0] = (expNode*) &var;
    }

    void expNode::setVarInfo(const int pos_, varInfo &var){
      const int pos = ((0 <= pos_) ? pos_ : leafCount);

      varInfo **varLeaves = (varInfo**) leaves[pos]->leaves;
      varInfo *&varLeaf   = varLeaves[0];

      varLeaf = &var;
    }

    typeInfo& expNode::getTypeInfo(){
      return *((typeInfo*) leaves[0]);
    }

    typeInfo& expNode::getTypeInfo(const int pos){
      typeInfo **typeLeaves = (typeInfo**) leaves[pos]->leaves;
      typeInfo *&typeLeaf   = typeLeaves[0];

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

    bool expNode::hasQualifier(const std::string &qualifier){
      if(info & expType::varInfo){
        return getVarInfo().hasQualifier(qualifier);
      }
      else if(info & expType::type){
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
      else if(info == expType::declaration){
        if(leafCount &&
           (leaves[0]->info & expType::type)){

          leaves[0]->changeType(newType);
        }
      }
    }

    int expNode::getVariableCount(){
      if(info == expType::declaration){
        return leafCount;
      }

      return 0;
    }

    bool expNode::variableHasInit(const int pos){
      if(info == expType::declaration){
        const expNode &varNode = *(getVariableNode(pos));

        return (varNode.leafCount &&
                (varNode.leaves[0]->value == "="));
      }

      return false;
    }

    expNode* expNode::getVariableNode(const int pos){
      if(info == expType::declaration){
        return leaves[pos];
      }

      return NULL;
    }

    expNode* expNode::getVariableInfoNode(const int pos){
      if(info == expType::declaration){
        expNode &varNode = *(getVariableNode(pos));

        expNode *varLeaf = ((varNode.info & expType::varInfo) ?
                            &varNode :
                            varNode.leaves[0]);

        if(varLeaf->info & expType::varInfo){
          return varLeaf;
        }
        else if(varNode.leafCount &&
                (varLeaf->value == "=")){

          return varLeaf->leaves[0];
        }
      }

      return NULL;
    }

    expNode* expNode::getVariableOpNode(const int pos){
      if(info == expType::declaration){
        expNode &varNode = *(getVariableNode(pos));

        if(varNode.leafCount &&
           (varNode[0].info == expType::LR)){

          return &(varNode[0]);
        }
      }

      return NULL;
    }

    expNode* expNode::getVariableInitNode(const int pos){
      if(info == expType::declaration){
        if(variableHasInit(pos)){
          const expNode &varNode = *(getVariableNode(pos));

          const expNode *varLeaf = ((varNode.info & expType::varInfo) ?
                                    &varNode :
                                    varNode.leaves[0]);

          if(varLeaf->value == "=")
            return varLeaf->leaves[1];
        }
      }

      return NULL;
    }

    std::string expNode::getVariableName(const int pos){
      if(info == expType::declaration){
        expNode &leaf = *(leaves[pos]);

        if(leaf.info & expType::varInfo){
          return leaf.getVarInfo().name;
        }
        else if(leaf.leafCount &&
                (leaf[0].value == "=")){

          return leaf[0].getVarInfo(0).name;
        }
      }

      return "";
    }

    int expNode::getUpdatedVariableCount(){
      if(leafCount == 0)
        return 0;

      expNode *cNode = leaves[0];
      int count = 0;

      while(cNode &&
            (cNode->value == ",")){

        if(2 <= cNode->leafCount)
          count += (isAnAssOperator((*cNode)[1].value));

        cNode = cNode->leaves[0];
      }

      if(cNode)
        count += isAnAssOperator(cNode->value);

      return count;
    }

    bool expNode::updatedVariableIsSet(const int pos){
      expNode *n = getUpdatedNode(pos);

      if(n == NULL)
        return false;

      return ((n->info & expType::LR) &&
              isAnAssOperator(n->value));
    }

    expNode* expNode::getUpdatedNode(const int pos){
      if(leafCount == 0)
        return NULL;

      int count = getUpdatedVariableCount();

      if(count <= pos)
        return NULL;

      expNode *cNode = leaves[0];

      while(cNode &&
            (cNode->value == ",")){

        if(2 <= cNode->leafCount)
          count -= (isAnAssOperator((*cNode)[1].value));

        if(pos == count)
          return cNode->leaves[1];

        cNode = cNode->leaves[0];
      }

      if(cNode){
        count -= isAnAssOperator(cNode->value);

        if(pos == count)
          return cNode;
      }

      return cNode;
    }

    expNode* expNode::getUpdatedVariableOpNode(const int pos){
      return getUpdatedNode(pos);
    }

    expNode* expNode::getUpdatedVariableInfoNode(const int pos){
      expNode *n = getUpdatedNode(pos);

      if(n == NULL)
        return NULL;

      return n->leaves[0];
    }

    expNode* expNode::getUpdatedVariableSetNode(const int pos){
      expNode *n = getUpdatedNode(pos);

      if(n == NULL)
        return NULL;

      return n->leaves[1];
    }

    int expNode::getVariableBracketCount(){
      if(info & expType::variable){
        if((1 < leafCount)                        &&
           ((*this)[1].info & expType::qualifier) &&
           (0 < (*this)[1].leafCount)             &&
           ((*this)[1][0].value == "[")){

          return (*this)[1].leafCount;
        }
      }

      return 0;
    }

    expNode* expNode::getVariableBracket(const int pos){
      if(info & expType::variable){
        if(pos < getVariableBracketCount())
          return &( (*this)[1][pos] );
      }

      return NULL;
    }

    //  ---[ Node-based ]----------
    std::string expNode::getMyVariableName(){
      if(info & expType::variable){
        if(leafCount == 0)
          return value;
        else
          return leaves[0]->getMyVariableName();
      }
      else if(info & expType::varInfo){
        return getVarInfo().name;
      }
      else if(info & expType::function){
        return value;
      }

      return "";
    }
    //  ===========================

    //  ---[ Statement-based ]-----
    void expNode::switchBaseStatement(statement &s1, statement &s2){
      expNode &flatRoot = *(makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if(n.hasVariable()){
          std::string varName = n.getMyVariableName();
          varInfo &var        = *(s1.hasVariableInScope(varName));

          statementNode *sn1 = &(s1.varUpdateMap[&var]);
          statementNode *sn2 = &(s1.varUsedMap[&var]);

          while(sn1){
            if(sn1->value == &s1)
              sn1->value = &s2;

            sn1 = sn1->right;
          }

          while(sn2){
            if(sn2->value == &s1)
              sn2->value = &s2;

            sn2 = sn2->right;
          }
        }
      }

      freeFlatHandle(flatRoot);
    }
    //  ===========================
    //================================


    //---[ Analysis Info ]------------
    bool expNode::valueIsKnown(const strToStrMap_t &stsMap){
      bool isKnown = true;

      expNode &flatRoot = *(makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if(n.info & (expType::varInfo   |
                     expType::unknown   |
                     expType::variable  |
                     expType::function)){ // [-] Check function later

          cStrToStrMapIterator it;

          if(n.info & expType::varInfo)
            it = stsMap.find(n.getVarInfo().name);
          else
            it = stsMap.find(n.value);

          if(it == stsMap.end()){
            isKnown = false;
            break;
          }
        }
        else if((n.info  == expType::C) && // [-] Don't load constant arrays yet
                (n.value == "[")){

          isKnown = false;
          break;
        }
      }

      freeFlatHandle(flatRoot);

      return isKnown;
    }

    typeHolder expNode::calculateValue(const strToStrMap_t &stsMap){
      if(valueIsKnown() == false)
        return typeHolder();

      expNode &this2 = *(clone());

      expNode &flatRoot = *(this2.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if(n.info & (expType::unknown  |
                     expType::variable |
                     expType::function | // [-] Check function later
                     expType::varInfo)){

          cStrToStrMapIterator it;

          if(n.info & expType::varInfo)
            it = stsMap.find(n.getVarInfo().name);
          else
            it = stsMap.find(n.value);

          n.info  = expType::presetValue;
          n.value = it->second;
        }
      }

      freeFlatHandle(flatRoot);

      return evaluateString(this2.toString());
    }
    //================================

    void expNode::freeLeaf(const int leafPos){
      leaves[leafPos]->free();
      delete leaves[leafPos];
    }

    void expNode::free(){
      // Let the parser free all varInfos
      if( !(info & expType::hasInfo) ){
        for(int i = 0; i < leafCount; ++i){
          leaves[i]->free();
          delete leaves[i];
        }

        if(leaves)
          delete [] leaves;
      }

      info      = 0;
      leafCount = 0;
    }

    void expNode::freeThis(){
      leafCount = 0;

      if(leaves)
        delete [] leaves;

      leaves = NULL;
    }

    void expNode::print(const std::string &tab){
      if( !(info & expType::hasInfo) ){

        std::cout << tab << "[" << getBits(info) << "] " << value << '\n';

        for(int i = 0; i < leafCount; ++i)
          leaves[i]->print(tab + "    ");
      }
      else if(info & expType::varInfo){
        if(info & expType::type)
          std::cout << tab << "[VT: " << getBits(info) << "] " << getVarInfo() << '\n';
        else
          std::cout << tab << "[V: " << getBits(info) << "] " << getVarInfo().name << '\n';
      }
      else if(info & expType::typeInfo){
        std::cout << tab << "[T: " << getBits(info) << "]\n" << getTypeInfo().toString(tab + "        ") << '\n';
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

      case (expType::LR):{
        if((value != ".") && (value != "->") && (value != ","))
          out << *(leaves[0]) << ' ' << value << ' ' << *(leaves[1]);
        else if(value == ",")
          out << *(leaves[0]) << value << ' ' << *(leaves[1]);
        else
          out << *(leaves[0]) << value << *(leaves[1]);

        break;
      }

      case (expType::LCR):{
        out << *(leaves[0]) << " ? " << *(leaves[1]) << " : " << *(leaves[2]);

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

      case (expType::varInfo | expType::declaration | expType::type):
      case (expType::varInfo | expType::type):{
        out << getVarInfo();

        break;
      }

      case (expType::varInfo | expType::declaration):{
        out << getVarInfo().toString(false);

        break;
      }

      case (expType::varInfo):{
        out << getVarInfo().name;

        break;
      }

      case (expType::typeInfo):{
        out << getTypeInfo().toString(tab) << ";\n";

        break;
      }

      case (expType::cast_):{
        out << '('
            << *(leaves[0])
            << ')';

        if(1 < leafCount)
          out << ' ' << *(leaves[1]);

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

      case (expType::return_):{
        out << tab;

        for(int i = 0; i < leafCount; ++i)
          out << *(leaves[i]);

        break;
      }

      case (expType::transfer_):{
        out << tab;

        out << value;

        if(leafCount){
          out << ' ';

          for(int i = 0; i < leafCount; ++i)
            out << *(leaves[i]);
        }

        out << ";\n";

        break;
      }

      case (expType::occaFor):{
        out << value << ' ';
        break;
      }

      case (expType::checkSInfo):{
        if(sInfo->info & flowStatementType){
          out << tab;

          if(sInfo->info & forStatementType)
            out << "for(";
          else if(sInfo->info & whileStatementType)
            out << "while(";
          else if(sInfo->info & ifStatementType){
            if(sInfo->info == ifStatementType)
              out << "if(";
            else if(sInfo->info == elseIfStatementType)
              out << "else if(";
            else
              out << "else";
          }
          else if(sInfo->info & switchStatementType)
            out << "switch(";

          if(leafCount){
            if(leaves[0]->info == expType::declaration)
              leaves[0]->printOn(out, "", (expFlag::noNewline |
                                           expFlag::noSemicolon));
            else
              out << *(leaves[0]);

            for(int i = 1; i < leafCount; ++i)
              out << "; " << *(leaves[i]);
          }

          if( !(sInfo->info & gotoStatementType) &&
              (sInfo->info != elseStatementType) ){
            out << ")";
          }
          else if(sInfo->info & gotoStatementType){
            out << ":";
          }
        }
        else if(sInfo->info & caseStatementType){
          const size_t tabChars = tab.size();

          if(2 < tabChars)
            out << tab.substr(0, tabChars - 2);

          if(leafCount)
            out << "case " << *(leaves[0]) << ':';
          else
            out << "default:";
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

    std::string expNode::toString(const int leafPos,
                                  const int printLeafCount){
      if(leafCount <= leafPos)
        return "";

      const int trueInfo = info;
      info = expType::root;

      int trueLeafCount = leafCount;

      leafCount = ((leafPos + printLeafCount <= leafCount) ?
                   (printLeafCount) :
                   (leafCount - leafPos));

      expNode **trueLeaves = leaves;
      leaves = new expNode*[leafCount];

      for(int i = 0; i < leafCount; ++i)
        leaves[i] = trueLeaves[leafPos + i];

      std::string ret = (std::string) *this;

      delete [] leaves;

      info = trueInfo;

      leaves    = trueLeaves;
      leafCount = trueLeafCount;

      return ret;
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
      varUpdateMap(pb.varUpdateMap),
      varUsedMap(pb.varUsedMap),

      depth(-1),
      info(blockStatementType),

      up(NULL),

      expRoot(*this),

      statementCount(0),
      statementStart(NULL),
      statementEnd(NULL) {}

    statement::statement(const int depth_,
                         varUsedMap_t &varUpdateMap_,
                         varUsedMap_t &varUsedMap_) :
      varUpdateMap(varUpdateMap_),
      varUsedMap(varUsedMap_),

      depth(depth_),
      info(blockStatementType),

      up(NULL),

      expRoot(*this),

      statementCount(0),
      statementStart(NULL),
      statementEnd(NULL) {}

    statement::statement(const int depth_,
                         const int info_,
                         statement *up_) :
      varUpdateMap(up_->varUpdateMap),
      varUsedMap(up_->varUsedMap),

      depth(depth_),
      info(info_),

      up(up_),

      expRoot(*this),

      statementCount(0),
      statementStart(NULL),
      statementEnd(NULL) {}

    statement::~statement(){};

    statement* statement::makeSubStatement(){
      return new statement(depth + 1,
                           0, this);
    }

    std::string statement::getTab(){
      std::string ret = "";
      statement *up_  = up;

      while(up_){
        ret += "  ";
        up_ = up_->up;
      }

      return ret;
    }

    void statement::labelStatement(strNode *&nodeRoot,
                                   expNode *expPtr,
                                   const bool parsingC){

      info = findStatementType(nodeRoot, expPtr, parsingC);
    }

    int statement::findStatementType(strNode *&nodeRoot,
                                     expNode *expPtr,
                                     const bool parsingC){
      if(!parsingC)
        return findFortranStatementType(nodeRoot, expPtr);

      if(nodeRoot->info == macroKeywordType)
        return checkMacroStatementType(nodeRoot, expPtr);

      else if(nodeRoot->info == 0)
        return 0;

      else if(nodeRoot->info == occaForType)
        return checkOccaForStatementType(nodeRoot, expPtr);

      else if((nodeRoot->info & typedefType) |
              (nodeRoot->info & structType))
        return checkStructStatementType(nodeRoot, expPtr);

      else if(nodeRoot->info & (operatorType |
                                presetValue))
        return checkUpdateStatementType(nodeRoot, expPtr);

      else if(nodeHasDescriptor(nodeRoot))
        return checkDescriptorStatementType(nodeRoot, expPtr);

      else if(nodeRoot->info & unknownVariable){
        if(nodeRoot->right &&
           nodeRoot->right->value == ":")
          return checkGotoStatementType(nodeRoot, expPtr);

        return checkUpdateStatementType(nodeRoot, expPtr);
      }

      else if(nodeRoot->info & flowControlType)
        return checkFlowStatementType(nodeRoot, expPtr);

      else if(nodeRoot->info & specialKeywordType)
        return checkSpecialStatementType(nodeRoot, expPtr);

      else if(nodeRoot->info & brace)
        return checkBlockStatementType(nodeRoot, expPtr);

      // Statement: (int) 3;
      else if(nodeRoot->info & parentheses)
        return checkUpdateStatementType(nodeRoot, expPtr);

      // Statement: [;]
      else if(nodeRoot->info & endStatement)
        return checkUpdateStatementType(nodeRoot, expPtr);

      else {
        while(nodeRoot &&
              !(nodeRoot->info & endStatement))
          nodeRoot = nodeRoot->right;

        return updateStatementType;
      }
    }

    int statement::findFortranStatementType(strNode *&nodeRoot,
                                            expNode *expPtr){

      if(nodeRoot->info == macroKeywordType)
        return checkMacroStatementType(nodeRoot, expPtr);

      else if(nodeRoot->info == 0)
        return 0;

      else if(nodeHasDescriptor(nodeRoot))
        return checkFortranDescriptorStatementType(nodeRoot, expPtr);

      else if(nodeRoot->info & unknownVariable)
        return checkFortranUpdateStatementType(nodeRoot, expPtr);

      else if(nodeRoot->info & flowControlType)
        return checkFortranFlowStatementType(nodeRoot, expPtr);

      else if(nodeRoot->info & specialKeywordType)
        return checkFortranSpecialStatementType(nodeRoot, expPtr);

      else {
        while(nodeRoot &&
              !(nodeRoot->info & endStatement))
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

      return occaForType;
    }

    int statement::checkStructStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(!typeInfo::statementIsATypeInfo(*this, nodeRoot))
        return checkDescriptorStatementType(nodeRoot);

      while(nodeRoot){
        if(nodeRoot->info & endStatement)
          break;

        nodeRoot = nodeRoot->right;
      }

      return structStatementType;
    }

    int statement::checkUpdateStatementType(strNode *&nodeRoot, expNode *expPtr){
      while(nodeRoot){
        if(nodeRoot->info & endStatement)
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
          if(nodeRoot->info & endStatement)
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

      OCCA_CHECK(false,
                 "You found the [Waldo 2] error in:\n"
                 << prettyString(nodeRoot, "  "));

      return 0;
    }

    int statement::checkSpecialStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(nodeRoot == NULL)
        return blankStatementType;

      const bool isCaseStatement = ((nodeRoot->value == "case") ||
                                    (nodeRoot->value == "default"));

      while(nodeRoot){
        if(nodeRoot->info & endStatement)
          break;

        nodeRoot = nodeRoot->right;
      }

      if(isCaseStatement)
        return caseStatementType;

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

    bool statement::nodeHasQualifier(strNode *n){
      if( !(n->info & qualifierType) )
        return false;

      // short and long can be both:
      //    specifiers and qualifiers
      if(n->info == keywordType["long"]){
        if((n->right) &&
           (hasTypeInScope(n->right->value))){

          return true;
        }
        else
          return false;
      }

      return true;
    }

    bool statement::nodeHasSpecifier(strNode *n){
      return ((n->info & specifierType) ||
              ((n->info & unknownVariable) &&
               ( hasTypeInScope(n->value) )));
    }

    bool statement::nodeHasDescriptor(strNode *n){
      if(nodeHasSpecifier(n) || nodeHasQualifier(n))
        return true;

      return false;
    }

    typeInfo* statement::hasTypeInScope(const std::string &typeName){
      cScopeTypeMapIterator it = scopeTypeMap.find(typeName);

      if(it != scopeTypeMap.end())
        return it->second;

      if(up)
        return up->hasTypeInScope(typeName);

      return NULL;
    }

    varInfo* statement::hasVariableInScope(const std::string &varName){
      const statement *sPos = this;

      while(sPos){
        cScopeVarMapIterator it = sPos->scopeVarMap.find(varName);

        if(it != sPos->scopeVarMap.end())
          return it->second;

        sPos = sPos->up;
      }

      return NULL;
    }

    varInfo* statement::hasVariableInLocalScope(const std::string &varName){
      scopeVarMapIterator it = scopeVarMap.find(varName);

      if(it != scopeVarMap.end())
        return it->second;

      return NULL;
    }

    bool statement::hasDescriptorVariable(const std::string descriptor){
      return hasQualifier(descriptor);
    }

    bool statement::hasDescriptorVariableInScope(const std::string descriptor){
      if(hasDescriptorVariable(descriptor))
        return true;

      if(up != NULL)
        return up->hasDescriptorVariable(descriptor);

      return false;
    }

    //---[ Loading ]--------------------
    void statement::loadAllFromNode(strNode *nodeRoot, const bool parsingC){
      while(nodeRoot)
        nodeRoot = loadFromNode(nodeRoot, parsingC);
    }

    strNode* statement::loadFromNode(strNode *nodeRoot, const bool parsingC){
      statement *newStatement = makeSubStatement();
      strNode * nodeRootEnd   = nodeRoot;

      newStatement->expRoot.loadFromNode(nodeRootEnd, parsingC);
      const int st = newStatement->info;

      OCCA_CHECK((st & invalidStatementType) == 0,
                 "Not a valid statement");

      if(st & skipStatementType){
        nodeRootEnd = skipAfterStatement(nodeRootEnd);

        delete newStatement;
        return nodeRootEnd;
      }

      addStatement(newStatement);

      if(st & simpleStatementType){
        nodeRootEnd = newStatement->loadSimpleFromNode(st,
                                                       nodeRoot,
                                                       nodeRootEnd,
                                                       parsingC);
      }

      else if(st & flowStatementType){
        if(st & forStatementType)
          nodeRootEnd = newStatement->loadForFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd,
                                                      parsingC);

        else if(st & whileStatementType)
          nodeRootEnd = newStatement->loadWhileFromNode(st,
                                                        nodeRoot,
                                                        nodeRootEnd,
                                                        parsingC);

        else if(st & ifStatementType)
          nodeRootEnd = loadIfFromNode(st,
                                       nodeRoot,
                                       nodeRootEnd,
                                       parsingC);

        else if(st & switchStatementType)
          nodeRootEnd = newStatement->loadSwitchFromNode(st,
                                                         nodeRoot,
                                                         nodeRootEnd,
                                                         parsingC);

        else if(st & gotoStatementType)
          nodeRootEnd = newStatement->loadGotoFromNode(st,
                                                       nodeRoot,
                                                       nodeRootEnd,
                                                       parsingC);
      }

      else if(st & caseStatementType)
        nodeRootEnd = newStatement->loadCaseFromNode(st,
                                                     nodeRoot,
                                                     nodeRootEnd,
                                                     parsingC);

      else if(st & blockStatementType)
        nodeRootEnd = newStatement->loadBlockFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd,
                                                      parsingC);

      else if(st & functionStatementType){
        if(st & functionDefinitionType)
          nodeRootEnd = newStatement->loadFunctionDefinitionFromNode(st,
                                                                     nodeRoot,
                                                                     nodeRootEnd,
                                                                     parsingC);

        else if(st & functionPrototypeType)
          nodeRootEnd = newStatement->loadFunctionPrototypeFromNode(st,
                                                                    nodeRoot,
                                                                    nodeRootEnd,
                                                                    parsingC);
      }

      else if(st & structStatementType)
        nodeRootEnd = newStatement->loadStructFromNode(st,
                                                       nodeRoot,
                                                       nodeRootEnd,
                                                       parsingC);

      else if(st & blankStatementType)
        nodeRootEnd = newStatement->loadBlankFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd,
                                                      parsingC);

      else if(st & macroStatementType)
        nodeRootEnd = newStatement->loadMacroFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd,
                                                      parsingC);

      // std::cout << "[" << getBits(newStatement->info) << "] s = " << *(newStatement) << '\n';

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
      strNode *nodeRoot = parserNS::splitContent(source);
      nodeRoot          = parserNS::labelCode(nodeRoot);

      expNode *ret = createExpNodeFrom(nodeRoot);

      free(nodeRoot);

      return ret;
    }

    expNode* statement::createPlainExpNodeFrom(const std::string &source){
      strNode *nodeRoot = parserNS::splitContent(source);
      nodeRoot          = parserNS::labelCode(nodeRoot);

      expNode *ret = new expNode(*this);
      ret->initLoadFromNode(nodeRoot);
      ret->initOrganization();

      free(nodeRoot);

      return ret;
    }

    strNode* statement::loadSimpleFromNode(const int st,
                                           strNode *nodeRoot,
                                           strNode *nodeRootEnd,
                                           const bool parsingC){

      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    strNode* statement::loadOneStatementFromNode(const int st,
                                                 strNode *nodeRoot,
                                                 strNode *nodeRootEnd,
                                                 const bool parsingC){

      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;

      if(parsingC){
        if(nodeRootEnd){
          if(nodeRootEnd->info == startBrace)
            loadAllFromNode(nodeRootEnd->down);
          else
            return loadFromNode(nodeRootEnd);
        }
      }
      else{
        return loadFromNode(nodeRootEnd, parsingC);
      }

      return nextNode;
    }

    strNode* statement::loadForFromNode(const int st,
                                        strNode *nodeRoot,
                                        strNode *nodeRootEnd,
                                        const bool parsingC){

      if(parsingC){
        return loadOneStatementFromNode(st,
                                        nodeRoot, nodeRootEnd,
                                        parsingC);
      }
      else {
        return loadUntilFortranEnd(nodeRootEnd);
      }
    }

    strNode* statement::loadWhileFromNode(const int st,
                                          strNode *nodeRoot,
                                          strNode *nodeRootEnd,
                                          const bool parsingC){

      if(parsingC){
        if(st == whileStatementType)
          return loadOneStatementFromNode(st, nodeRoot, nodeRootEnd);
        else{
          strNode *nextNode = loadOneStatementFromNode(st, nodeRootEnd, nodeRootEnd);
          info = whileStatementType;

          expRoot.loadFromNode(nextNode);

          info = doWhileStatementType;

          // Skip the [;] after [while()]
          if(nextNode &&
             (nextNode->value == ";")){

            nextNode = nextNode->right;
          }

          return nextNode;
        }
      }
      else{
        return loadUntilFortranEnd(nodeRootEnd);
      }
    }

    strNode* statement::loadIfFromNode(const int st_,
                                       strNode *nodeRoot,
                                       strNode *nodeRootEnd,
                                       const bool parsingC){

      statement *newStatement = statementEnd->value;

      if(parsingC){
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

            OCCA_CHECK((st & invalidStatementType) == 0,
                       "Not a valid statement");

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
      else{
        if(nodeRoot->value != "IF")
          return newStatement->loadUntilFortranEnd(nodeRootEnd);

        strNode *nodePos = nodeRoot;

        while(nodePos != nodeRootEnd){
          if(nodePos->value == "THEN")
            return newStatement->loadUntilFortranEnd(nodeRootEnd);

          nodePos = nodePos->right;
        }

        // [IF][(...)][load this]
        newStatement->loadFromNode(nodeRoot->right->right, parsingFortran);

        return nodeRootEnd;
      }
    }

    // [-] Missing Fortran
    strNode* statement::loadSwitchFromNode(const int st,
                                           strNode *nodeRoot,
                                           strNode *nodeRootEnd,
                                           const bool parsingC){

      if(parsingC){
        return loadOneStatementFromNode(st,
                                        nodeRoot, nodeRootEnd,
                                        parsingC);
      }
      else {
        return loadUntilFortranEnd(nodeRootEnd);
      }
    }

    // [-] Missing Fortran
    strNode* statement::loadCaseFromNode(const int st,
                                           strNode *nodeRoot,
                                           strNode *nodeRootEnd,
                                           const bool parsingC){

      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    // [-] Missing Fortran
    strNode* statement::loadGotoFromNode(const int st,
                                         strNode *nodeRoot,
                                         strNode *nodeRootEnd,
                                         const bool parsingC){

      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    strNode* statement::loadFunctionDefinitionFromNode(const int st,
                                                       strNode *nodeRoot,
                                                       strNode *nodeRootEnd,
                                                       const bool parsingC){

      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      if(parsingC){
        if(nodeRootEnd)
          loadAllFromNode(nodeRootEnd->down);

        return nextNode;
      }
      else
        return loadUntilFortranEnd(nextNode);
    }

    // [-] Missing Fortran
    strNode* statement::loadFunctionPrototypeFromNode(const int st,
                                                      strNode *nodeRoot,
                                                      strNode *nodeRootEnd,
                                                      const bool parsingC){

      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    // [-] Missing Fortran
    strNode* statement::loadBlockFromNode(const int st,
                                          strNode *nodeRoot,
                                          strNode *nodeRootEnd,
                                          const bool parsingC){

      strNode *nextNode = nodeRoot->right;

      if(nodeRoot->down)
        loadAllFromNode(nodeRoot->down);

      return nextNode;
    }

    // [-] Missing Fortran
    strNode* statement::loadStructFromNode(const int st,
                                           strNode *nodeRoot,
                                           strNode *nodeRootEnd,
                                           const bool parsingC){

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
                                          strNode *nodeRootEnd,
                                          const bool parsingC){

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
                                          strNode *nodeRootEnd,
                                          const bool parsingC){

      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    //  ---[ Fortran ]--------
    // [+] Missing
    int statement::checkFortranStructStatementType(strNode *&nodeRoot, expNode *expPtr){
      nodeRoot = skipUntilFortranStatementEnd(nodeRoot);

      return structStatementType;
    }

    int statement::checkFortranUpdateStatementType(strNode *&nodeRoot, expNode *expPtr){
      nodeRoot = skipUntilFortranStatementEnd(nodeRoot);

      return updateStatementType;
    }

    int statement::checkFortranDescriptorStatementType(strNode *&nodeRoot, expNode *expPtr){
      if((nodeRoot        && (nodeRoot->value        == "IMPLICIT")) &&
         (nodeRoot->right && (nodeRoot->right->value == "NONE"))){

        nodeRoot = skipUntilFortranStatementEnd(nodeRoot);

        return skipStatementType;
      }

      varInfo var;
      nodeRoot = var.loadFromFortran(*this, nodeRoot);

      if( !(var.info & varType::functionDef) ){
        nodeRoot = skipUntilFortranStatementEnd(nodeRoot);
      }

      if(var.info & varType::var)
        return declareStatementType;
      else
        return functionDefinitionType;
    }

    int statement::checkFortranFlowStatementType(strNode *&nodeRoot, expNode *expPtr){
      if(expPtr)
        expPtr->info  = expType::checkSInfo;

      std::string &nodeValue = nodeRoot->value;

      int st = 0;

      if(nodeValue == "DO")
        st = forStatementType;
      else if(nodeValue == "DO WHILE")
        st = whileStatementType;
      else if(nodeValue == "IF")
        st = ifStatementType;
      else if(nodeValue == "ELSE IF")
        st = elseIfStatementType;
      else if(nodeValue == "ELSE")
        st = elseStatementType;
      else if(nodeValue == "SWITCH")
        st = switchStatementType;

      // [-] Missing one-line case
      while(nodeRoot &&
            (nodeRoot->value != "\\n") &&
            (nodeRoot->value != ";")){

        nodeRoot = nodeRoot->right;
      }

      if(nodeRoot)
        nodeRoot = nodeRoot->right;

      if(st)
        return st;

      OCCA_CHECK(false,
                 "You found the [Waldo 3] error in:\n"
                 << prettyString(nodeRoot, "  "));

      return 0;
    }

    int statement::checkFortranSpecialStatementType(strNode *&nodeRoot, expNode *expPtr){
      strNode *nextNode = skipUntilFortranStatementEnd(nodeRoot);

      if(nodeRoot->value == "CALL"){
        nodeRoot = nextNode;

        return updateStatementType;
      }
      else if((nodeRoot->value == "FUNCTION") ||
              (nodeRoot->value == "SUBROUTINE")){

        return checkFortranDescriptorStatementType(nodeRoot, expPtr);
      }

      nodeRoot = nextNode;

      return blankStatementType;
    }

    bool statement::isFortranEnd(strNode *nodePos){
      return (nodePos == getFortranEnd(nodePos));
    }

    strNode* statement::getFortranEnd(strNode *nodePos){
      if(info & functionDefinitionType){
        const std::string &typeName = (getFunctionVar()->baseType->name);
        const std::string endTag    = ((typeName == "void") ?
                                       "ENDSUBROUTINE" : "ENDFUNCTION");

        return skipNodeUntil(nodePos, endTag);
      }
      else if(info & (forStatementType |
                      whileStatementType)){

        return skipNodeUntil(nodePos, "ENDDO");
      }
      else if(info & ifStatementType){
        if(info != elseStatementType){
          int pos1, pos2, pos3;

          strNode *node1 = skipNodeUntil(nodePos, "ENDIF"  , &pos1);
          strNode *node2 = skipNodeUntil(nodePos, "ELSE IF", &pos2);
          strNode *node3 = skipNodeUntil(nodePos, "ELSE"   , &pos3);

          if(pos1 < pos2){
            if(pos1 < pos3)
              return node1;
            else
              return node3;
          }
          else{
            if(pos2 < pos3)
              return node2;
            else
              return node3;
          }
        }
        else
          return skipNodeUntil(nodePos, "ENDIF");
      }

      return nodePos;
    }

    strNode* statement::getFortranEnd(strNode *nodePos,
                                      const std::string &value){
      if((value == "DO") ||
         (value == "DO WHILE")){

        return skipNodeUntil(nodePos, "ENDDO");
      }
      else if((value == "IF") ||
              (value == "ELSE IF")){
        int pos1, pos2, pos3;

        strNode *node1 = skipNodeUntil(nodePos, "ENDIF"  , &pos1);
        strNode *node2 = skipNodeUntil(nodePos, "ELSE IF", &pos2);
        strNode *node3 = skipNodeUntil(nodePos, "ELSE"   , &pos3);

        if(pos1 < pos2){
          if(pos1 < pos3)
            return node1;
          else
            return node3;
        }
        else{
          if(pos2 < pos3)
            return node2;
          else
            return node3;
        }
      }
      else if(value == "ELSE"){
        return skipNodeUntil(nodePos, "ENDIF");
      }
      else if(value == "FUNCTION"){
        return skipNodeUntil(nodePos, "ENDFUNCTION");
      }
      else if(value == "SUBROUTINE"){
        return skipNodeUntil(nodePos, "ENDSUBROUTINE");
      }

      return nodePos;
    }

    strNode* statement::loadUntilFortranEnd(strNode *nodePos){
      while(!isFortranEnd(nodePos))
        nodePos = loadFromNode(nodePos, parsingFortran);

      // Don't skip [ELSE IF] and [ELSE]
      if(nodePos &&
         (nodePos->value.substr(0,3) == "END")){

        nodePos = skipAfterStatement(nodePos);
      }

      return nodePos;
    }

    strNode* statement::skipNodeUntil(strNode *nodePos,
                                      const std::string &value,
                                      int *separation){
      int count = 0;

      while(nodePos &&
            (nodePos->value != value)){

        ++count;
        nodePos = nodePos->right;
      }

      if(separation)
        *separation = count;

      return nodePos;
    }

    strNode* statement::skipAfterStatement(strNode *nodePos){
      nodePos = skipUntilStatementEnd(nodePos);

      if(nodePos)
        nodePos = nodePos->right;

      return nodePos;
    }

    strNode* statement::skipUntilStatementEnd(strNode *nodePos){
      while(nodePos){
        if(nodePos->info & endStatement)
          break;

        nodePos = nodePos->right;
      }

      return nodePos;
    }

    strNode* statement::skipUntilFortranStatementEnd(strNode *nodePos){
      while(nodePos){
        nodePos = nodePos->right;

        if((nodePos->value == "\\n") ||
           (nodePos->value == ";")){

          break;
        }
      }

      return nodePos;
    }
    //==================================

    statement* statement::getGlobalScope(){
      statement *globalScope = this;

      while(globalScope->up)
        globalScope = globalScope->up;

      return globalScope;
    }

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
      newS->info      = type_;

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
      newS->info      = type_;

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

    statement& statement::createStatementFromSource(const std::string &source){
      statementNode sn;

      pushSourceRightOf(&sn, source);

      statement &ret = *(sn.right->value);

      delete sn.right;

      return ret;
    }

    void statement::addStatementFromSource(const std::string &source){
      loadFromNode(labelCode( splitContent(source) ));
    }

    void statement::addStatementsFromSource(const std::string &source){
      loadAllFromNode(labelCode( splitContent(source) ));
    }

    void statement::pushSourceLeftOf(statementNode *target,
                                     const std::string &source){
      addStatementFromSource(source);

      statementNode *newSN = statementEnd;

      statementEnd        = statementEnd->left;
      statementEnd->right = NULL;

      if(statementStart == target)
        statementStart = newSN;

      if(target->left)
        target->left->right = newSN;

      newSN->left  = target->left;
      newSN->right = target;

      target->left = newSN;
    }

    void statement::pushSourceRightOf(statementNode *target,
                                      const std::string &source){
      addStatementFromSource(source);

      if(target == statementEnd->left)
        return;

      statementNode *newSN = statementEnd;

      statementEnd        = statementEnd->left;
      statementEnd->right = NULL;

      if(target->right)
        target->right->left = newSN;

      newSN->right  = target->right;
      newSN->left   = target;

      target->right = newSN;
    }

    //---[ Misc ]---------------------
    bool statement::hasBarrier(){
      expNode &flatRoot = *(expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        if(flatRoot[i].value == "occaBarrier")
          return true;
      }

      expNode::freeFlatHandle(flatRoot);

      return false;
    }

    bool statement::hasStatementWithBarrier(){
      if(hasBarrier())
        return true;

      statementNode *statementPos = statementStart;

      while(statementPos){
        statement &s = *(statementPos->value);

        if(s.hasBarrier())
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    // Guaranteed to work with statements under a globalScope
    statement& statement::greatestCommonStatement(statement &s){
      std::vector<statement*> path[2];

      for(int pass = 0; pass < 2; ++pass){
        statement *cs = ((pass == 0) ? this : &s);

        while(cs){
          path[pass].push_back(cs);
          cs = cs->up;
        }
      }

      const int dist0   = (int) path[0].size();
      const int dist1   = (int) path[1].size();
      const int minDist = ((dist0 < dist1) ? dist0 : dist1);

      for(int i = 1; i <= minDist; ++i){
        if(path[0][dist0 - i] != path[1][dist1 - i])
          return *(path[0][dist0 - i + 1]);
      }

      return *(path[0][dist0 - minDist]);
    }

    unsigned int statement::distToForLoop(){
      return distToStatementType(forStatementType);
    }

    unsigned int statement::distToOccaForLoop(){
      statement *s = this;

      unsigned int dist = 0;

      while(s){
        if((s->info == occaForType) ||
           ((s->info == forStatementType) &&
            (s->getForStatementCount() == 4))){

          return dist;
        }

        s = s->up;
        ++dist;
      }

      return -1; // Maximum distance
    }

    unsigned int statement::distToStatementType(const int info_){
      statement *s = this;

      unsigned int dist = 0;

      while(s){
        if(s->info == info_)
          return dist;

        s = s->up;
        ++dist;
      }

      return -1; // Maximum distance
    }

    bool statement::insideOf(statement &s){
      statement *up_ = up;

      while(up_){
        if(up_ == &s)
          return true;

        up_ = up_->up;
      }

      return false;
    }

    void statement::setStatementIdMap(statementIdMap_t &idMap){
      int startID = 0;

      setStatementIdMap(idMap, startID);
    }

    void statement::setStatementIdMap(statementIdMap_t &idMap,
                                      int &startID){

      statementNode *nodePos = statementStart;

      while(nodePos){
        statement &s = *(nodePos->value);
        idMap[&s] = startID++;

        s.setStatementIdMap(idMap, startID);

        nodePos = nodePos->right;
      }
    }

    void statement::setStatementVector(statementVector_t &vec,
                                       const bool init){

      statementNode *nodePos = statementStart;

      if(init)
        vec.clear();

      while(nodePos){
        statement &s = *(nodePos->value);

        vec.push_back(&s);

        s.setStatementVector(vec, false);

        nodePos = nodePos->right;
      }
    }

    void statement::setStatementVector(statementIdMap_t &idMap,
                                       statementVector_t &vec){

      statementIdMapIterator it = idMap.begin();

      const int statementCount_ = idMap.size();

      vec.clear();
      vec.resize(statementCount_);

      for(int i = 0; i < statementCount_; ++i){
        vec[ it->second ] = (it->first);

        ++it;
      }
    }

    void statement::removeFromUpdateMapFor(varInfo &var){
      removeFromMapFor(var, varUpdateMap);
    }

    void statement::removeFromUsedMapFor(varInfo &var){
      removeFromMapFor(var, varUsedMap);
    }

    void statement::removeFromMapFor(varInfo &var,
                                     varUsedMap_t &usedMap){
      varUsedMapIterator it = usedMap.find(&var);

      if(it == usedMap.end())
        return;

      statementNode *sn = &(it->second);

      while(sn->value == this){
        if(sn->right != NULL){
          statementNode *snRight = sn->right;

          sn->value = snRight->value;
          sn->right = snRight->right;

          delete snRight;
        }
        else{
          sn->value = NULL;
          return;
        }
      }

      sn = sn->right;

      while(sn){
        if(sn->value == this)
          popAndGoRight(sn);
        else
          sn = sn->right;
      }
    }
    //================================

    void statement::checkIfVariableIsDefined(varInfo &var,
                                             statement *origin){
      if(var.name.size() == 0)
        return;

      scopeVarMapIterator it = scopeVarMap.find(var.name);

      OCCA_CHECK((it == scopeVarMap.end())  ||
                 var.hasQualifier("extern") ||
                 (var.info & varType::functionDef),

                 "Variable [" << var.name << "] defined in:\n"
                 << *origin
                 << "is already defined in:\n"
                 << *this);
    }

    statement* statement::getVarOriginStatement(varInfo &var){
      varUsedMapIterator it = varUpdateMap.find(&var);

      if(it == varUpdateMap.end())
        return NULL;

      return (it->second).value;
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
      if(var->name.size() == 0)
        return;

      checkIfVariableIsDefined(*var, origin);

      scopeVarMap[var->name] = var;

      addVariableToUpdateMap(*var, origin);
    }

    void statement::addVariableToUpdateMap(varInfo &var,
                                           statement *origin_){

      statement *origin = (origin_ == NULL ? this : origin_);

      addVariableToMap(var, varUpdateMap, origin);
    }

    void statement::addVariableToUsedMap(varInfo &var,
                                         statement *origin_){

      statement *origin = (origin_ == NULL ? this : origin_);

      addVariableToMap(var, varUsedMap, origin);
    }

    void statement::addVariableToMap(varInfo &var,
                                     varUsedMap_t &usedMap,
                                     statement *origin){
      statementNode &sn = usedMap[&var];

      if(sn.value)
        lastNode(&sn)->push(origin);
      else
        sn.value = origin;
    }

    void statement::addStatement(statement *newStatement){
      newStatement->up    = this;
      newStatement->depth = (depth + 1);

      if(statementStart != NULL){
        ++statementCount;
        statementEnd = statementEnd->push(newStatement);
      }
      else{
        statementCount = 1;
        statementStart = new statementNode(newStatement);
        statementEnd   = statementStart;
      }
    }

    void statement::removeStatement(statement &s){
      statementNode *sn = statementStart;

      if(sn == NULL)
        return;

      if(sn->value == &s){
        statementStart = statementStart->right;

        if(sn == statementEnd)
          statementEnd = NULL;

        delete sn;

        return;
      }

      while(sn){
        if(sn->value == &s){
          if(sn == statementEnd)
            statementEnd = statementEnd->left;

          delete sn->pop();

          return;
        }

        sn = sn->right;
      }
    }

    statement* statement::clone(statement *up_){
      statement *newStatement;

      if(up_){
        newStatement = new statement(up_->depth + 1,
                                     info, up_);
      }
      else if(up){
        newStatement = new statement(depth,
                                     info, up);
      }
      else {
        newStatement = new statement(depth,
                                     varUpdateMap,
                                     varUsedMap);
      }

      expRoot.cloneTo(newStatement->expRoot);

      newStatement->statementStart = NULL;
      newStatement->statementEnd   = NULL;

      statementNode *sn = statementStart;

      while(sn){
        newStatement->addStatement( sn->value->clone(newStatement) );
        sn = sn->right;
      }

      // Add ninja-variables (nin-nin)
      scopeVarMapIterator it = scopeVarMap.begin();

      while(it != scopeVarMap.end()){
        varInfo &var = *(it->second);

        if(!newStatement->hasVariableInLocalScope(var.name))
          newStatement->addVariable(var);

        ++it;
      }

      return newStatement;
    }

    void statement::printVariablesInScope(){
      if(up)
        up->printVariablesInScope();

      printVariablesInLocalScope();
    }

    void statement::printVariablesInLocalScope(){
      scopeVarMapIterator it = scopeVarMap.begin();

      while(it != scopeVarMap.end()){
        std::cout << "  " << *(it->second) << '\n';

        ++it;
      }
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
    void statement::createUniqueVariables(std::vector<std::string> &names,
                                          const int flags){
      std::stringstream ss;

      const int nameCount = names.size();
      int iterCount = 0;

      while(true){
        if(flags & statementFlag::updateByNumber)
          ss << iterCount++;
        else
          ss << "_";

        const std::string &suffix = ss.str();

        for(int i = 0; i < nameCount; ++i){
          if(hasVariableInScope(names[i] + suffix))
            break;

          if((i + 1) == nameCount){
            for(int j = 0; j < nameCount; ++j)
              names[j] += suffix;

            return;
          }
        }

        if(flags & statementFlag::updateByNumber)
          ss.str("");
      }
    }

    void statement::createUniqueSequentialVariables(std::string &varName,
                                                    const int varCount){
      std::stringstream ss;

      // Find unique baseName
      while(true){
        int v;

        for(v = 0; v < varCount; ++v){
          ss << v;

          if(hasVariableInLocalScope(varName + ss.str()))
            break;

          ss.str("");
        }

        if(v == varCount)
          break;

        varName += '_';
      }
    }

    void statement::swapExpWith(statement &s){
      expNode::swap(expRoot, s.expRoot);
    }

    bool statement::hasQualifier(const std::string &qualifier){
      if(info & declareStatementType){
        varInfo &var = getDeclarationVarInfo(0);
        return var.hasQualifier(qualifier);
      }
      else if(info & functionStatementType){
        varInfo &var = expRoot.getVarInfo(0);
        return var.hasQualifier(qualifier);
      }
      else if(info & forStatementType){
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

      if(info & declareStatementType){
        varInfo &var = getDeclarationVarInfo(0);
        var.addQualifier(qualifier);
      }
      else if(info & functionStatementType){
        varInfo &var = expRoot.getVarInfo(0);
        var.addQualifier(qualifier, pos);
      }
      // ---[ OLD ]---
      // else if(info & forStatementType){
      //   if(expRoot.leafCount){
      //     expNode &node1    = *(expRoot.leaves[0]);
      //     expNode &qualNode = *(node1.leaves[0]);

      //     if( !(qualNode.leaves[0]->info & expType::qualifier) )
      //       qualNode.addNode(expType::qualifier, 0);

      //     qualNode.leaves[0]->value = qualifier;
      //   }
      // }
    }

    void statement::removeQualifier(const std::string &qualifier){
      if(!hasQualifier(qualifier))
        return;

      if(info & declareStatementType){
        varInfo &var = getDeclarationVarInfo(0);
        var.removeQualifier(qualifier);
      }
      else if(info & functionStatementType){
      }
      else if(info & forStatementType){
      }
    }


    int statement::occaForInfo(){
      if(info != occaForType)
        return notAnOccaFor;

      std::string forLoop = expRoot.value;
      const int chars     = forLoop.size();

      const int nest = (1 + forLoop[chars - 1] - '0');

      if(forLoop.find("Inner") != std::string::npos)
        return (nest << occaInnerForShift);

      return (nest << occaOuterForShift);
    }

    int statement::occaForNest(const int forInfo){
      if(forInfo & occaInnerForMask)
        return ((forInfo >> occaInnerForShift) - 1);

      return ((forInfo >> occaOuterForShift) - 1);
    }

    bool statement::isOccaOuterFor(const int forInfo){
      return ((forInfo & occaOuterForMask) != 0);
    }

    bool statement::isOccaInnerFor(const int forInfo){
      return ((forInfo & occaInnerForMask) != 0);
    }

    void statement::setVariableDeps(varInfo &var,
                                    sDep_t &sDep){

      expNode &flatRoot = *(expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if(n.hasVariable()){
          std::string nVarName = n.getMyVariableName();
          varInfo *nVar        = hasVariableInScope(nVarName);

          // [-] Missing up-checks
          //    Example: var->x = 3
          //                    =
          //              ->        3
          //            var  x
          if((nVar != &var) || // Checking our variable update
             (n.up == NULL) || // Update needs an assignment operator
             !isAnAssOperator(n.up->value) ||
             (n.up->leaves[0]->getMyVariableName() != var.name)){

            continue;
          }

          // Get right-side of assignment operator
          expNode &exp = *(n.up->leaves[1]);

          addVariableDeps(exp, sDep);
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void statement::addVariableDeps(expNode &exp,
                                    sDep_t &sDep){
      if(exp.leafCount == 0){
        if(exp.hasVariable()){
          varInfo &var = *(hasVariableInScope(exp.value));

          sDep.uniqueAdd(var);
        }

        return;
      }

      expNode &flatRoot = *(exp.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if(n.hasVariable()){
          varInfo &var = *(hasVariableInScope(n.value));

          sDep.uniqueAdd(var);
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    bool statement::setsVariableValue(varInfo &var){
      expNode &flatRoot = *(expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if(n.hasVariable()){
          std::string nVarName = n.getMyVariableName();
          varInfo *nVar        = hasVariableInScope(nVarName);

          // [-] Missing up-checks
          if((nVar != &var) || // Checking our variable update
             (n.up == NULL) || // Update needs an assignment operator
             (n.up->value != "=") ||
             (n.up->leaves[0]->getMyVariableName() != var.name)){

            continue;
          }

          expNode::freeFlatHandle(flatRoot);

          return true;
        }
      }

      expNode::freeFlatHandle(flatRoot);

      return false;
    }

    void statement::addStatementDependencies(statementIdMap_t &idMap,
                                             statementVector_t sVec,
                                             idDepMap_t &depMap){

      addStatementDependencies(*this, idMap, sVec, depMap);
    }

    void statement::addStatementDependencies(statement &fromS,
                                             statementIdMap_t &idMap,
                                             statementVector_t sVec,
                                             idDepMap_t &depMap){
      if(info & functionStatementType)
        return;

      expNode &flatRoot = *(expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if(n.hasVariable()){
          std::string varName = n.getMyVariableName();
          varInfo *pVar = fromS.hasVariableInScope(varName);

          if(pVar){
            varDepGraph vdg(*pVar, fromS, idMap);
            vdg.addFullDependencyMap(depMap, idMap, sVec);
          }
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void statement::addNestedDependencies(statementIdMap_t &idMap,
                                          statementVector_t sVec,
                                          idDepMap_t &depMap){

      statementNode *sn = statementStart;

      while(sn){
        sn->value->addStatementDependencies(idMap, sVec, depMap);
        sn->value->addNestedDependencies(idMap, sVec, depMap);

        sn = sn->right;
      }
    }

    varInfo& statement::getDeclarationVarInfo(const int pos){
      expNode *varNode = expRoot.getVariableInfoNode(pos);
      return varNode->getVarInfo();
    }

    expNode* statement::getDeclarationVarNode(const int pos){
      if(info & declareStatementType)
        return expRoot.leaves[pos];

      return NULL;
    }

    std::string statement::getDeclarationVarName(const int pos){
      if(info & declareStatementType){
        varInfo &var = getDeclarationVarInfo(pos);
        return var.name;
      }

      return "";
    }

    expNode* statement::getDeclarationVarInitNode(const int pos){
      if(info & declareStatementType)
        return expRoot.getVariableInitNode(pos);

      return NULL;
    }

    int statement::getDeclarationVarCount(){
      if(info & declareStatementType)
        return expRoot.leafCount;

      return 0;
    }

    varInfo* statement::getFunctionVar(){
      if(info & functionStatementType){
        return &(expRoot.getVarInfo(0));
      }
      else if(info & updateStatementType){
        statement *s = up;

        while(s &&
              !(s->info & functionStatementType)){
          s = s->up;
        }

        if(s)
          return s->getFunctionVar();

        return NULL;
      }

      OCCA_CHECK(false, "Not added yet");

      return NULL;
    }

    void statement::setFunctionVar(varInfo &var){
      if(info & functionStatementType){
        expRoot.setVarInfo(0, var);
      }
      else if(info & updateStatementType){
        statement *s = up;

        while(s &&
              !(s->info & functionStatementType)){
          s = s->up;
        }

        if(s)
          s->setFunctionVar(var);
      }
    }

    std::string statement::getFunctionName(){
      if(info & functionStatementType){
        return getFunctionVar()->name;
      }

      OCCA_CHECK(false, "Not added yet");

      return "";
    }

    void statement::setFunctionName(const std::string &newName){
      if(info & functionStatementType){
        getFunctionVar()->name = newName;
        return;
      }

      OCCA_CHECK(false, "Not added yet");
    }

    bool statement::functionHasQualifier(const std::string &qName){
      if(info & functionStatementType){
        return getFunctionVar()->hasQualifier(qName);
      }

      OCCA_CHECK(false, "Not added yet");

      return false;
    }

    int statement::getFunctionArgCount(){
      if(info & functionStatementType){
        return getFunctionVar()->argumentCount;
      }

      return 0;
    }

    std::string statement::getFunctionArgType(const int pos){
      if(info & functionDefinitionType){
        return getFunctionVar()->baseType->name;
      }

      return "";
    }

    std::string statement::getFunctionArgName(const int pos){
      if(info & functionDefinitionType){
        return getFunctionVar()->getArgument(pos).name;
      }

      return "";
    }

    varInfo* statement::getFunctionArgVar(const int pos){
      if(info & functionDefinitionType){
        return &(getFunctionVar()->getArgument(pos));
      }

      return NULL;
    }

    bool statement::hasFunctionArgVar(varInfo &var){
      if(info & functionDefinitionType){
        const int argc = getFunctionArgCount();

        for(int i = 0; i < argc; ++i){
          if(&var == getFunctionArgVar(i))
            return true;
        }

        return false;
      }

      return false;
    }

    void statement::addFunctionArg(const int pos, varInfo &var){
      if( !(info & functionStatementType) )
        return;

      getFunctionVar()->addArgument(pos, var);
    }

    expNode* statement::getForStatement(const int pos){
      if(info & forStatementType)
        return expRoot.leaves[pos];

      return NULL;
    }

    int statement::getForStatementCount(){
      if(info & forStatementType)
        return expRoot.leafCount;

      return 0;
    }
    //================================

    // autoMode: Handles newlines and tabs
    std::string statement::prettyString(strNode *nodeRoot,
                                        const std::string &tab_,
                                        const bool autoMode){
      return "";
#if 0
      strNode *nodePos = nodeRoot;

      std::string tab = tab_;
      std::string ret = "";

      while(nodePos){
        if(nodePos->info & operatorType){

          if(nodePos->info & binaryOperatorType){

            // char *blah
            if(nodeHasQualifier(nodePos)){

              // [char ][*][blah]
              // or
              // [int ][a][ = ][0][, ][*][b][ = ][1][;]
              //                       ^
              if(nodePos->left &&
                 ((nodePos->left->info & descriptorType) ||
                  (nodePos->left->value == ","))){
                ret += *nodePos;

                // [const ][*][ const]
                if(nodePos->right &&
                   (nodePos->right->info & descriptorType) &&
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
            else if(nodePos->info & unitaryOperatorType){
              // (-blah ... )
              if(nodePos->left &&
                 !(nodePos->left->info & (presetValue | unknownVariable)) )
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
                  !(nodePos->left->info & unknownVariable)) ||
                 ((nodePos->right == NULL) ||
                  !(nodePos->right->info & unknownVariable))){
                if(nodePos->left){
                  nodePos->up->print();
                  std::cout << "1. Error on:\n";
                  nodePos->left->print("  ");
                }
                else{
                  std::cout << "2. Error on:\n";
                  nodePos->print("  ");
                }

                OCCA_THROW;
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
          else if(nodePos->info & unitaryOperatorType){
            ret += *nodePos;
          }
          else if(nodePos->info & ternaryOperatorType){
            ret += " ? ";

            nodePos = nodePos->right;

            OCCA_CHECK((nodePos->right) != NULL,
                       "3. Error on: " << *(nodePos->left));

            if((nodePos->down).size())
              ret += prettyString(nodePos, "", autoMode);
            else
              ret += *nodePos;

            ret += " : ";

            OCCA_CHECK(((nodePos->right)        != NULL) &&
                       ((nodePos->right->right) != NULL),

                       "4. Error on: " << *(nodePos->left->left));

            nodePos = nodePos->right->right;

            if((nodePos->down).size())
              ret += prettyString(nodePos, "", autoMode);
            else
              ret += *nodePos;
          }
        }

        else if(nodePos->info & brace){
          if(nodePos->info & startSection){
            // a[] = {};
            if(nodePos->up->info & binaryOperatorType){
              ret += "{ ";
            }
            else{
              // Case: function(...) const {
              if( (((nodePos->sideDepth) != 0) &&
                   ((nodePos->up->down[nodePos->sideDepth - 1]->info & parentheses) ||
                    (nodePos->up->down[nodePos->sideDepth - 1]->value == "const")) )

                  || (nodePos->up->info & (occaKeywordType | flowControlType)))
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
               (nodePos->up->info & binaryOperatorType))
              ret += " }";
            else{
              ret += '}';

              //   }
              // }
              if((nodePos->up == NULL) ||
                 ((nodePos->up->right) &&
                  (nodePos->up->right->info == endBrace)))
                ret += "\n" + tab.substr(0, tab.size() - 2);
              else
                ret += "\n" + tab;
            }
          }
        }

        else if(nodePos->info == endParentheses){
          ret += ")";

          // if(...) statement
          if(autoMode)
            if((nodePos->up->info & flowControlType) &&
               (((nodePos->sideDepth) >= (nodePos->up->down.size() - 1)) ||
                !(nodePos->up->down[nodePos->sideDepth + 1]->info & brace))){

              ret += "\n" + tab + "  ";
            }
        }

        else if(nodePos->info & endStatement){
          ret += *nodePos;

          // for(){
          //   ...;
          // }
          if((nodePos->right == NULL) ||
             ((nodePos->right) &&
              (nodePos->right->info & brace))){

            ret += "\n" + tab.substr(0, tab.size() - 2);
          }
          //   blah;
          // }
          else if(!(nodePos->up)                    ||
                  !(nodePos->up->info & flowControlType) ||
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
             ((nodePos->right->info & (presetValue    |
                                       unknownVariable)) ||
              nodeHasDescriptor(nodePos->right))){

            ret += " ";
          }
        }

        else if(nodePos->info & flowControlType){
          ret += *nodePos;

          if(autoMode)
            if(nodePos->down.size() == 0)
              ret += '\n' + tab + "  ";
        }

        else if(nodePos->info & specialKeywordType){
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
        else if(nodePos->info & macroKeywordType){
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

    std::string statement::toString(const int flags){
      std::string tab;

      if(flags & statementFlag::printSubStatements)
        tab = getTab();

      statementNode *statementPos = statementStart;

      // OCCA For's
      if(info == occaForType){
        if( !(flags & statementFlag::printSubStatements) )
          return expRoot.value;

        std::string ret = tab + expRoot.value + "{\n";

        while(statementPos){
          ret += (std::string) *(statementPos->value);
          statementPos = statementPos->right;
        }

        ret += tab + "}\n";

        return ret;
      }

      else if(info & declareStatementType){
        return expRoot.toString(tab);
      }

      else if(info & (simpleStatementType | gotoStatementType)){
        if(flags & statementFlag::printSubStatements)
          return expRoot.toString(tab) + "\n";
        else
          return expRoot.toString();
      }

      else if(info & flowStatementType){
        if( !(flags & statementFlag::printSubStatements) )
          return expRoot.toString();

        std::string ret;

        if(info != doWhileStatementType){
          ret += expRoot.toString(tab);

          if(statementCount > 1)
            ret += "{";
          else if(statementCount == 0) // The [Jesse Chan] Case
            ret += "\n" + tab + "  ;";

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
           (info == doWhileStatementType)){

            ret += tab + "}\n";
        }

        if(info == doWhileStatementType){
          ret += ' ';
          ret += expRoot.toString();
          ret += ";\n\n";
        }

        return ret;
      }

      else if(info & caseStatementType){
        if(flags & statementFlag::printSubStatements)
          return expRoot.toString(tab) + "\n";
        else
          return expRoot.toString();
      }

      else if(info & functionStatementType){
        if(info & functionDefinitionType){
          if( !(flags & statementFlag::printSubStatements) )
            return expRoot.toString();

          std::string ret = expRoot.toString(tab);

          ret += " {\n";

          while(statementPos){
            ret += (std::string) *(statementPos->value);
            statementPos = statementPos->right;
          }

          if(back(ret) != '\n')
            ret += tab + "\n}\n\n";
          else
            ret += tab + "}\n\n";

          return ret;
        }
        else if(info & functionPrototypeType)
          return expRoot.toString(tab);
      }
      else if(info & blockStatementType){
        if( !(flags & statementFlag::printSubStatements) )
          return "{}";

        std::string ret = "";

        if(0 <= depth)
          ret += tab + "{\n";

        while(statementPos){
          ret += (std::string) *(statementPos->value);
          statementPos = statementPos->right;
        }

        if(0 <= depth){
          if(back(ret) != '\n')
            ret += "\n" + tab + "}\n";
          else
            ret += tab + "}\n";
        }

        return ret;
      }
      else if(info & structStatementType){
        if(flags & statementFlag::printSubStatements)
          return expRoot.toString(tab) + "\n";
        else
          return expRoot.toString();
      }
      else if(info & macroStatementType){
        if(flags & statementFlag::printSubStatements)
          return tab + expRoot.value + "\n";
        else
          return expRoot.value;
      }

      return expRoot.toString(tab);
    }

    std::string statement::onlyThisToString(){
      return toString(statementFlag::printEverything &
                      ~statementFlag::printSubStatements);
    }

    statement::operator std::string() {
      return toString();
    }

    std::ostream& operator << (std::ostream &out, statement &s){
      out << (std::string) s;

      return out;
    }
    //============================================

    bool isAnOccaTag(const std::string &tag){
      return (isAnOccaInnerTag(tag) ||
              isAnOccaOuterTag(tag));
    }

    bool isAnOccaInnerTag(const std::string &tag){
      if( (tag.find("inner") == std::string::npos) ||
          ((tag != "inner0") &&
           (tag != "inner1") &&
           (tag != "inner2")) ){

        return false;
      }

      return true;
    }

    bool isAnOccaOuterTag(const std::string &tag){
      if( (tag.find("outer") == std::string::npos) ||
          ((tag != "outer0") &&
           (tag != "outer1") &&
           (tag != "outer2")) ){

        return false;
      }

      return true;
    }
  };
};
