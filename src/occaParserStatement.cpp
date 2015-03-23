#include "occaParserStatement.hpp"
#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
    //---[ Exp Node ]-------------------------------
    expNode::expNode() :
      sInfo(NULL),

      value(""),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL),

      type(NULL) {}

    expNode::expNode(const std::string &str) :
      sInfo(NULL),

      value(str),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL),

      type(NULL) {}

    expNode::expNode(const char *c) :
      sInfo(NULL),

      value(c),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL),

      type(NULL) {}

    expNode::expNode(const expNode &e) :
      sInfo(e.sInfo),

      value(e.value),
      info(e.info),

      up(e.up),

      leafCount(e.leafCount),
      leaves(e.leaves),

      type(e.type) {}

    expNode::expNode(statement &s) :
      sInfo(&s),

      value(""),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL),

      type(NULL) {}

    expNode& expNode::operator = (const expNode &e){
      sInfo = e.sInfo;

      value = e.value;
      info  = e.info;

      up = e.up;

      leafCount = e.leafCount;
      leaves    = e.leaves;

      type = e.type;

      return *this;
    }

    expNode expNode::makeFloatingLeaf(){
      expNode fLeaf;

      fLeaf.sInfo = sInfo;
      fLeaf.up    = this;

      return fLeaf;
    }

    int expNode::getStatementType(){
      if(info & expType::macro_)
        return smntType::macroStatement;

      else if(info & expType::occaFor)
        return keywordType["occaOuterFor0"];

      else if(info & (expType::qualifier |
                      expType::type)){

        if(typeInfo::statementIsATypeInfo(*this, 0))
          return smntType::structStatement;

        varInfo var;
        var.loadFrom(*this, 0);

        if(var.info & varType::var)
          return smntType::declareStatement;
        else if(var.info & varType::functionDec)
          return smntType::functionPrototype;
        else
          return smntType::functionDefinition;
      }

      else if((info & (expType::unknown |
                       expType::variable)) &&
              (1 < leafCount) &&
              (leaves[1]->value == ":")){

        return smntType::gotoStatement;
      }

      else if((info == expType::C) &&
              (leaves[0]->value == "{")){

        return smntType::blockStatement;
      }

      return smntType::updateStatement;
    }

    void expNode::loadFromNode(expNode &allExp,
                               const bool parsingC){
      int expPos = 0;

      loadFromNode(allExp, expPos, parsingC);
    }

    void expNode::loadFromNode(expNode &allExp,
                               int &expPos,
                               const bool parsingC){

      if(allExp.leafCount <= expPos){
        sInfo->info = smntType::invalidStatement;
        return;
      }

      int expStart = expPos;

      sInfo->labelStatement(allExp, expPos, parsingC);

      // [<>] Make sure expPos returns the guy after our last leaf
      useExpLeaves(allExp, expStart, (expPos - expStart));

      // Don't need to load stuff
      if((sInfo->info & (smntType::skipStatement   |
                         smntType::macroStatement  |
                         smntType::gotoStatement   |
                         smntType::blockStatement))      ||
         (sInfo->info == smntType::occaFor)              ||
         (sInfo->info == smntType::elseStatement)        ||
         (sInfo->info == smntType::doWhileStatement)){

        return;
      }

      //---[ Special Type ]---
      if((*this)[0].info & preExpType::specialKeyword){
        if(((*this)[0].value == "break")    ||
           ((*this)[0].value == "continue")){

          if(((*this)[0].value == "continue") &&
             (sInfo->distToOccaForLoop() <= sInfo->distToForLoop())){

            value = "occaContinue";
            info  = expType::transfer_;
          }
          else{
            value = (*this)[0].value;
            info  = expType::transfer_;
          }

          return;
        }

        // [-] Doesn't support GCC's twisted [Labels as Values]
        if((*this)[0].value == "goto"){
          OCCA_CHECK(1 < leafCount,
                     "Goto check [" << toString() << "] needs label");

          value = allExp[expStart + 1];
          info  = expType::goto_;
          return;
        }

        // Case where nodeRoot = [case, return]

        if(((*this)[0].value == "case") ||
           ((*this)[0].value == "default")){
          info = expType::checkSInfo;
        }
        else if((*this)[0].value == "return"){
          info = expType::return_;
        }
      }
      //======================

      if(parsingC)
        splitAndOrganizeNode();
      else
        splitAndOrganizeFortranNode();

      std::cout << "[" << getBits(sInfo->info) << "] this = " << *this << '\n';
      print();
    }

    void expNode::splitAndOrganizeNode(){
      changeExpTypes();
      printf("Printing after changeExpTypes()\n");
      print();
      initOrganization();

      if(sInfo->info & smntType::declareStatement)
        splitDeclareStatement();

      else if(sInfo->info & smntType::updateStatement)
        splitUpdateStatement();

      else if((sInfo->info & (smntType::ifStatement    |
                              smntType::forStatement   |
                              smntType::whileStatement |
                              smntType::switchStatement)) &&
              (sInfo->info != smntType::elseStatement)){

        splitFlowStatement();
      }

      else if(sInfo->info & smntType::functionStatement)
        splitFunctionStatement();

      else if(sInfo->info & smntType::structStatement)
        splitStructStatement();

      else if(sInfo->info & smntType::caseStatement)
        splitCaseStatement();

      else
        organize();
    }

    void expNode::splitAndOrganizeFortranNode(){
      changeFortranExpTypes();

      if(leaves[leafCount - 1]->value == "\\n")
        --leafCount;

      if(sInfo->info & smntType::declareStatement)
        splitFortranDeclareStatement();

      if(sInfo->info & smntType::updateStatement)
        splitFortranUpdateStatement();

      else if((sInfo->info & (smntType::ifStatement  |
                              smntType::forStatement |
                              smntType::whileStatement)) &&
              (sInfo->info != smntType::elseStatement)){

        splitFortranFlowStatement();
      }

      else if(sInfo->info & smntType::functionStatement)
        splitFortranFunctionStatement();

      else if(sInfo->info & smntType::structStatement)
        splitStructStatement();

      else if(sInfo->info & smntType::caseStatement)
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
          leaf.addNodes(0, 1, sExpEnd - sExpStart);

          for(int j = sExpStart; j < sExpEnd; ++j)
            expNode::swap(*leaf.leaves[j - sExpStart + 1], *leaves[j]);

          leaf.organizeLeaves();
        }

        if(leafPos < leafCount)
          removeNode(leafPos);
      }

      expNode::swap(*this, newExp);
    }

    void expNode::splitUpdateStatement(){
      info = expType::checkSInfo;

      int updateCount = 1 + typeInfo::delimiterCount(*this, ",");
      int leafPos     = 0;

      // Store variables and stuff
      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(expType::root, 0, updateCount);

      for(int i = 0; i < updateCount; ++i){
        expNode &leaf = newExp[i];

        int sExpStart = leafPos;
        int sExpEnd   = typeInfo::nextDelimiter(*this, leafPos, ",");

        leafPos = (sExpEnd + 1);

        // Don't put the [;]
        if((sExpEnd == leafCount) &&
           (leaves[sExpEnd - 1]->value == ";")){

          --sExpEnd;
        }

        if(sExpStart < sExpEnd){
          leaf.addNodes(0, 0, sExpEnd - sExpStart);

          for(int j = sExpStart; j < sExpEnd; ++j)
            expNode::swap(*(leaf.leaves[j - sExpStart]), *leaves[j]);

          leaf.organizeLeaves();
        }
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
          leaf.addNodes(0, 0, (nextLeafPos - leafPos));

          for(int j = 0; j < leaf.leafCount; ++j){
            delete leaf.leaves[j];

            leaf.leaves[j]     = expDown.leaves[leafPos + j];
            leaf.leaves[j]->up = &leaf;
          }

          if(!(sInfo->info & smntType::forStatement) || (i != 0))
            leaf.organize();
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
      if(sInfo->info & smntType::functionDefinition)
        info = (expType::funcInfo | expType::declaration);
      else
        info = (expType::funcInfo | expType::prototype);

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

      if(info == (expType::funcInfo | expType::declaration)){
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
          leaf.addNodes(0, 1, sExpEnd - sExpStart);

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

        sInfo->info = smntType::skipStatement;
      }
      else{ // Add variables to scope
        for(int i = 0; i < varCount; ++i){

          varInfo &var = leaves[i]->getVarInfo(0);
          varInfo *pVar = sInfo->hasVariableInScope(var.name);

          // Check if it's a function argument
          if(pVar != NULL){
            statement *s = sInfo->getVarOriginStatement(*pVar);

            if(s &&
               (s->info & smntType::functionDefinition)){

              // Hide stack info in arguments
              var.stackPointersUsed = 0;

              // Make sure it registers as a pointer
              if((var.pointerCount      == 0) &&
                 (var.stackPointerCount != 0)){

                var.pointerCount = 1;
                var.rightQualifiers.add("*", 0);
              }

              *(pVar) = var;

              sInfo->info = smntType::skipStatement;
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
          sInfo->info = smntType::skipStatement;
          return;
        }

        if(sInfo->hasVariableInScope(leaves[1]->value)){
          removeNode(0);

          leaves[0]->info = expType::funcInfo;

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

      leaves[0] = new expNode( makeFloatingLeaf() );
      leaves[1] = retValueLeaf;

      (*this)[0].info  = expType::printValue;
      (*this)[0].value = "return";

      addNode(expType::operator_, ";");
    }

    void expNode::splitFortranFlowStatement(){
      info = expType::checkSInfo;

      if(leafCount == 0)
        return;

      if(sInfo->info & smntType::forStatement){
        splitFortranForStatement();
      }
      // [IF/ELSE IF/DO WHILE]( EXPR )
      else if((sInfo->info == smntType::ifStatement)     ||
              (sInfo->info == smntType::elseIfStatement) ||
              (sInfo->info == smntType::whileStatement)){

        OCCA_CHECK(leafCount != 0,
                   "No expression in if-statement: " << *this << '\n');

        leaves[0]       = leaves[1];
        leaves[0]->info = expType::root;
        leaves[0]->organize();

        leafCount = 1;
      }
      // [ELSE]
      else if(sInfo->info & smntType::elseStatement){
        if(leafCount)
          free();
      }
    }

    void expNode::splitFortranForStatement(){
      // [DO] iter=start,end[,stride][,loop]
      // Infinite [DO]
      if(leafCount == 1){
        leaves[0]->value = "true";
        leaves[0]->info  = preExpType::presetValue;
        leafCount = 1;

        sInfo->info = smntType::whileStatement;

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

      newExp[0] = sInfo->createExpNodeFrom(iter + " = " + doStart);

      if(statementCount == 3){
        newExp[1] = sInfo->createExpNodeFrom("0 <= (" + doStrideSign + "* (" + doEnd + " - " + iter + "))");
        newExp[2] = sInfo->createExpNodeFrom(iter + " += " + doStride);
      }
      else{
        newExp[1] = sInfo->createExpNodeFrom(iter + " <= " + doEnd);
        newExp[2] = sInfo->createExpNodeFrom("++" + iter);
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
      info = (expType::funcInfo | expType::declaration);

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

    void expNode::translateOccaKeyword(expNode &exp, const bool parsingC){
      if(exp.info & preExpType::occaKeyword){

        if(((parsingC)  &&
            (exp.value == "directLoad")) ||
           ((!parsingC) &&
            upStringCheck(exp.value, "DIRECTLOAD"))){

          exp.value = "occaDirectLoad";
        }

      }
    }

    void expNode::changeExpTypes(){
      if(leafCount == 0)
        return;

      for(int leafPos = 0; leafPos < leafCount; ++leafPos){
        expNode &leaf = *(leaves[leafPos]);

        if(leaf.info & preExpType::occaKeyword)
          translateOccaKeyword(leaf, true);

        if(leaf.info & preExpType::unknownVariable){
          varInfo *nodeVar = sInfo->hasVariableInScope(leaf.value);

          if(nodeVar){
            if( !(nodeVar->info & varType::functionType) ){
              leaf.putVarInfo(*nodeVar);
            }
            else
              leaf.info = expType::funcInfo; // [<>] Change to funcInfo
          }
          else{
            typeInfo *nodeType = sInfo->hasTypeInScope(leaf.value);

            if(!nodeType)
              leaf.info = expType::unknown;
            else
              leaf.info = expType::type;
          }
        }

        else if(leaf.info & preExpType::presetValue){
          leaf.info = expType::presetValue;
        }

        else if(leaf.info & preExpType::descriptor){
          if(leaf.info == keywordType["long"]){
            if(((leafPos + 1) < leafCount) &&
               (sInfo->hasTypeInScope(leaves[leafPos + 1]->value))){

              leaf.info = expType::qualifier;
            }
            else
              leaf.info = expType::type;
          }
          else if(leaf.info & (preExpType::qualifier | preExpType::struct_))
            leaf.info = expType::qualifier;
          else
            leaf.info = expType::type;

          // For [*] and [&]
          if(leaf.info & preExpType::operator_)
            leaf.info |= expType::operator_;
        }

        else if(leaf.info & preExpType::struct_){
          leaf.info = expType::qualifier;
        }

        else if(leaf.info & preExpType::operator_){
          leaf.info = expType::operator_;
        }

        else if(leaf.info & preExpType::startSection){
          leaf.info = expType::C;

          if(leaf.leafCount)
            leaf.changeExpTypes();
        }

        else
          leaf.info = expType::printValue;

        if(leaf.info == 0){
          removeNode(leafPos);
          --leafPos;
        }
      }
    }

    void expNode::changeFortranExpTypes(){
      if(leafCount == 0)
        return;

      for(int leafPos = 0; leafPos < leafCount; ++leafPos){
        expNode &leaf = *(leaves[leafPos]);

        if(leaf.info & preExpType::unknownVariable){
          varInfo *nodeVar = sInfo->hasVariableInScope(leaf.value);

          if(nodeVar)
            leaf.info = expType::variable;
          else
            leaf.info = expType::unknown;
        }

        else if(leaf.info & preExpType::presetValue){
          leaf.info = expType::presetValue;
        }

        else if(leaf.info & preExpType::descriptor){
          if(leaf.info & preExpType::qualifier)
            leaf.info = expType::qualifier;
          else
            leaf.info  = expType::type;
        }

        else if(leaf.info & preExpType::operator_){
          leaf.info = expType::operator_;
        }

        else if(leaf.info & preExpType::startSection){
          leaf.info  = expType::C;

          if(leaf.leafCount)
            leaf.changeFortranExpTypes();
        }

        else
          leaf.info = expType::printValue;

        if(leaf.info == 0){
          removeNode(leafPos);
          --leafPos;
        }
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

        if(levelType & preExpType::unitaryOperator){
          bool updateNow = true;

          // Cases:  1 + [-]1
          //         (+1)
          if(!(leaves[leafPos]->info & expType::hasInfo) &&
             (leaves[leafPos]->value.size() == 1)        &&
             ((leaves[leafPos]->value[0] == '+') ||
              (leaves[leafPos]->value[0] == '-') ||
              (leaves[leafPos]->value[0] == '*') ||
              (leaves[leafPos]->value[0] == '&'))){

            if(leafPos &&
               ((leaves[leafPos - 1]->leafCount != 0) ||
                !(leaves[leafPos - 1]->info & expType::operator_))){

              updateNow = false;
            }
          }

          if(updateNow){
            int target = leafPos + ((levelType & preExpType::lUnitaryOperator) ?
                                    1 : -1);

            if((target < 0) || (leafCount <= target)){
              ++leafPos;
            }
            else{
              if(levelType & preExpType::lUnitaryOperator)
                leafPos = mergeLeftUnary(leafPos);
              else
                leafPos = mergeRightUnary(leafPos);
            }
          }
          else
            ++leafPos;
        }
        else if(levelType & preExpType::binaryOperator)
          leafPos = mergeBinary(leafPos);
        else if(levelType & preExpType::ternaryOperator)
          leafPos = mergeTernary(leafPos);
        else
          ++leafPos;
      }
    }

    int expNode::mergeRange(const int newLeafType,
                            const int leafPosStart,
                            const int leafPosEnd){
      expNode *newLeaf = new expNode( makeFloatingLeaf() );

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
                (sInfo->info == smntType::declareStatement)){

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
      if(sInfo->info & smntType::functionStatement)
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
             (leaves[leafPos - 1]->info & expType::funcInfo)){
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
            newLeaf = new expNode( makeFloatingLeaf() );

            newLeaf->up        = this;
            newLeaf->info      = expType::variable;
            newLeaf->leafCount = 2;
            newLeaf->leaves    = new expNode*[2];
          }

          expNode *sNewLeaf = new expNode( newLeaf->makeFloatingLeaf() );

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
    int expNode::mergeLeftUnary(const int leafPos){
      if(leafCount <= (leafPos + 1))
        return leafPos + 1;

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
      if(0 == leafPos)
        return leafPos + 1;

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
      if((0 == leafPos) || (leafCount <= (leafPos + 1)))
        return leafPos + 1;

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
      if((0 == leafPos) || (leafCount <= (leafPos + 3)))
        return leafPos + 1;

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
           (leaves[leafPos - 1]->info & (expType::variable | //   something
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
          n.info      = expType::funcInfo;
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

    expNode expNode::clone(){
      expNode newExp;
      newExp.sInfo = sInfo;

      cloneTo(newExp);

      return newExp;
    }

    expNode expNode::clone(statement &s){
      expNode newExp(s);

      cloneTo(newExp);

      return newExp;
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
      const bool isFuncInfo = ((info == (expType::funcInfo |
                                         expType::declaration)) ||
                               (info == (expType::funcInfo |
                                         expType::prototype)));

      const bool inForStatement = ((newExp.sInfo != NULL) &&
                                   (newExp.sInfo->info & smntType::forStatement));

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
            newExp.leaves[i] = new expNode( newExp.makeFloatingLeaf() );
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
      if(info & expType::hasInfo)
        return 0;

      int ret = leafCount;

      for(int i = 0; i < leafCount; ++i){
        if(leaves[i]->leafCount)
          ret += leaves[i]->nestedLeafCount();
      }

      return ret;
    }

    expNode& expNode::lastNode(){
      return *(leaves[leafCount - 1]);
    }

    expNode* expNode::makeFlatHandle(){
      expNode *flatNode;

      if(sInfo != NULL)
        flatNode = new expNode(*sInfo);
      else
        flatNode = new expNode( makeFloatingLeaf() );

      if(leafCount == 0)
        return flatNode;

      flatNode->info   = expType::printLeaves;
      flatNode->leaves = new expNode*[nestedLeafCount()];

      int offset = 0;
      makeFlatHandle(offset, flatNode->leaves);

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
      if(flatRoot.leafCount)
        delete [] flatRoot.leaves;

      delete &flatRoot;
    }

    void expNode::addNodes(const int info_,
                           const int pos_,
                           const int count){

      const int pos = ((0 <= pos_) ? pos_ : leafCount);

      reserveAndShift(pos, count);

      for(int i = pos; i < (pos + count); ++i){
        leaves[i] = new expNode( makeFloatingLeaf() );

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

    int expNode::insertExpAfter(expNode &exp, int pos){
      reserveAndShift(pos, exp.leafCount);

      for(int i = pos; i < (pos + exp.leafCount); ++i)
        leaves[i] = exp.leaves[i - pos];

      return (pos + exp.leafCount);
    }

    void expNode::useExpLeaves(expNode &exp, const int pos, const int count){
      reserveAndShift(0, count);

      for(int i = pos; i < (pos + count); ++i)
        leaves[i] = exp.leaves[i - pos];
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

    varInfo& expNode::addVarInfoNode(){
      addNode(0);

      varInfo **varLeaves = (varInfo**) leaves;
      varInfo *&varLeaf   = varLeaves[0];

      varLeaf = new varInfo();
      return *varLeaf;
    }

    varInfo& expNode::addVarInfoNode(const int pos){
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
                 expType::funcInfo)){

        if( (info & expType::varInfo) ||
            (value.size() != 0) ){

          return true;
        }
      }

      return false;
    }

    varInfo& expNode::getVarInfo(){
      return *((varInfo*) leaves[0]);
    }

    varInfo& expNode::getVarInfo(const int pos){
      varInfo **varLeaves = (varInfo**) leaves[pos]->leaves;
      varInfo *&varLeaf   = varLeaves[0];

      return *varLeaf;
    }

    void expNode::setVarInfo(varInfo &var){
      leaves[0] = (expNode*) &var;
    }

    void expNode::setVarInfo(const int pos, varInfo &var){
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

    void expNode::removeNodes(int pos, const int count){
      if(pos < 0)
        pos += leafCount;

      int removed = (((pos + count) <= leafCount) ?
                     count : (leafCount - pos));

      for(int i = (pos + removed); i < leafCount; ++i)
        leaves[i - count] = leaves[i];

      leafCount -= removed;
    }

    void expNode::removeNode(const int pos){
      removeNodes(pos, 1);
    }

    void expNode::convertTo(const int info_){
      if(info == expType::declaration){
        if(info_ & expType::variable){
          info = expType::variable;

          leafCount = 2;

          expNode *varNode = new expNode((*this)[1][0].clone(*sInfo));

          leaves[1]->free();
          leaves[1] = varNode;
        }
      }
    }

    bool expNode::hasQualifier(const std::string &qualifier){
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
      // ---[ OLD ]---
      if(info & expType::variable){
        if(leafCount){
          expNode &lqNode = *(leaves[0]);

          OCCA_CHECK((lqNode.info & expType::type) != 0,
                     "5. Error on:" << *this);

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
      // ---[ OLD ]---
      if(info & expType::variable){
        if(leafCount){
          expNode &lqNode = *(leaves[0]);

          OCCA_CHECK((lqNode.info & expType::type) != 0,
                     "5. Error on:" << *this);

          if( !(lqNode.lastLeaf()->info & expType::qualifier) )
            lqNode.addNode(expType::qualifier);

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
      else if(info == expType::declaration){
        if(leafCount &&
           (leaves[0]->info & expType::type)){

          leaves[0]->changeType(newType);
        }
      }
    }

    int expNode::getVariableCount(){
      if((info == expType::declaration)   ||
         ((info & expType::checkSInfo) &&
          (sInfo->info & smntType::updateStatement))){

        return leafCount;
      }

      return 0;
    }

    bool expNode::variableHasInit(const int pos){
      if((info == expType::declaration)   ||
         ((info & expType::checkSInfo) &&
          (sInfo->info & smntType::updateStatement))){

        const expNode &varNode = *(getVariableNode(pos));

        return (varNode.leafCount &&
                (varNode.leaves[0]->value == "="));
      }

      return false;
    }

    expNode* expNode::getVariableNode(const int pos){
      if((info == expType::declaration)   ||
         ((info & expType::checkSInfo) &&
          (sInfo->info & smntType::updateStatement))){

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
                (varLeaf->info == expType::LR)){

          return varLeaf->leaves[0]->getVariableInfoNode(0);
        }
      }
      else if(info & expType::varInfo){
        return this;
      }
      // else if(info & expType::variable){
      //   return this;
      // }

      return NULL;
    }

    expNode* expNode::getVariableInitNode(const int pos){
      if((info == expType::declaration)   ||
         ((info & expType::checkSInfo) &&
          (sInfo->info & smntType::updateStatement))){

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

    expNode* expNode::getVariableRhsNode(const int pos){
      if((info == expType::declaration)   ||
         ((info & expType::checkSInfo) &&
          (sInfo->info & smntType::updateStatement))){

        if(variableHasInit(pos)){
          const expNode &varNode = *(getVariableNode(pos));

          const expNode *varLeaf = ((varNode.info & expType::varInfo) ?
                                    &varNode :
                                    varNode.leaves[0]);

          if(varLeaf->info == expType::LR)
            return varLeaf->leaves[1];
        }
      }

      return NULL;
    }

    std::string expNode::getVariableName(const int pos){
      expNode *varNode = getVariableInfoNode(pos);

      if(varNode != NULL)
        return varNode->getVarInfo().name;

      return "";
    }

    //  ---[ Node-based ]----------
    std::string expNode::getMyVariableName(){
      if(info & expType::variable){
        if(leafCount == 0)
          return value;
        else
          return leaves[0]->value; // a[0] -> {a, [ {0}}
      }
      else if(info & expType::varInfo){
        return getVarInfo().name;
      }
      else if(info & expType::funcInfo){
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

        if(n.info & (expType::unknown  |
                     expType::variable |
                     expType::funcInfo | // [-] Check function later
                     expType::varInfo)){

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
      }

      freeFlatHandle(flatRoot);

      return isKnown;
    }

    typeHolder expNode::calculateValue(const strToStrMap_t &stsMap){
      if(valueIsKnown() == false)
        return typeHolder();

      expNode this2 = clone();

      expNode &flatRoot = *(this2.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &n = flatRoot[i];

        if(n.info & (expType::unknown  |
                     expType::variable |
                     expType::funcInfo | // [-] Check function later
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

      return evaluateExpression(this2);
    }
    //================================

    void expNode::freeLeaf(const int leafPos){
      leaves[leafPos]->free();
      delete leaves[leafPos];
    }

    // [-] Not properly done for varInfo and typeInfo
    void expNode::free(){
      // Let the parser free all varInfos
      if(info & expType::hasInfo){
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
        if((value != ".") && (value != "->"))
          out << *(leaves[0]) << ' ' << value << ' ' << *(leaves[1]);
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

      case (expType::funcInfo | expType::prototype):{
        if(leafCount)
          out << tab << getVarInfo(0) << ";\n";

        break;
      }

      case (expType::funcInfo | expType::declaration):{
        if(leafCount)
          out << tab << getVarInfo(0);

        break;
      }

      case (expType::funcInfo):{
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
          out << type.toString(tab) << ';';
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
        out << tab << "goto " << value << ";\n";
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
        if(sInfo->info & smntType::updateStatement){
          if(leafCount){
            leaves[0]->printOn(out, tab, (expFlag::noNewline |
                                          expFlag::noSemicolon));

            for(int i = 1; i < leafCount; ++i){
              out << ", ";

              leaves[i]->printOn(out, "", (expFlag::noNewline |
                                           expFlag::noSemicolon));
            }

            out << ";";
          }

          break;
        }

        else if(sInfo->info & smntType::flowStatement){
          out << tab;

          if(sInfo->info & smntType::forStatement)
            out << "for(";
          else if(sInfo->info & smntType::whileStatement)
            out << "while(";
          else if(sInfo->info & smntType::ifStatement){
            if(sInfo->info == smntType::ifStatement)
              out << "if(";
            else if(sInfo->info == smntType::elseIfStatement)
              out << "else if(";
            else
              out << "else";
          }
          else if(sInfo->info & smntType::switchStatement)
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

          if( !(sInfo->info & smntType::gotoStatement) &&
              (sInfo->info != smntType::elseStatement) ){
            out << ")";
          }
          else if(sInfo->info & smntType::gotoStatement){
            out << ":";
          }
        }
        else if(sInfo->info & smntType::caseStatement){
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
      info(smntType::blockStatement),

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
      info(smntType::blockStatement),

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

      for(int i = 0; i < depth; ++i)
        ret += "  ";

      return ret;
    }

    //---[ Find Statement ]-------------
    void statement::labelStatement(expNode &allExp,
                                   int &expPos,
                                   const bool parsingC){

      info = findStatementType(allExp, expPos, parsingC);
    }

    int statement::findStatementType(expNode &allExp,
                                     int &expPos,
                                     const bool parsingC){
      if(!parsingC)
        return findFortranStatementType(allExp, expPos);

      if(allExp[expPos].info == preExpType::macroKeyword)
        return checkMacroStatementType(allExp, expPos);

      else if(allExp[expPos].info == 0)
        return 0;

      else if(allExp[expPos].info == keywordType["occaOuterFor0"])
        return checkOccaForStatementType(allExp, expPos);

      else if((allExp[expPos].info & preExpType::typedef_) |
              (allExp[expPos].info & preExpType::struct_))
        return checkStructStatementType(allExp, expPos);

      else if(allExp[expPos].info & (preExpType::operator_ |
                                      preExpType::presetValue))
        return checkUpdateStatementType(allExp, expPos);

      else if(expHasDescriptor(allExp, expPos))
        return checkDescriptorStatementType(allExp, expPos);

      else if(allExp[expPos].info & preExpType::unknownVariable){
        if(((expPos + 1) < allExp.leafCount) &&
           (allExp[expPos + 1].value == ":")){

          return checkGotoStatementType(allExp, expPos);
        }

        return checkUpdateStatementType(allExp, expPos);
      }

      else if(allExp[expPos].info & preExpType::flowControl)
        return checkFlowStatementType(allExp, expPos);

      else if(allExp[expPos].info & preExpType::specialKeyword)
        return checkSpecialStatementType(allExp, expPos);

      else if(allExp[expPos].info & preExpType::brace)
        return checkBlockStatementType(allExp, expPos);

      // Statement: (int) 3;
      else if(allExp[expPos].info & preExpType::parentheses)
        return checkUpdateStatementType(allExp, expPos);

      // Statement: [;]
      else if(allExp[expPos].info & preExpType::endStatement)
        return checkUpdateStatementType(allExp, expPos);

      else {
        while((expPos < allExp.leafCount) &&
              !(allExp[expPos].info & preExpType::endStatement)){

          ++expPos;
        }

        return smntType::updateStatement;
      }
    }

    int statement::findFortranStatementType(expNode &allExp,
                                            int &expPos){

      if(allExp[expPos].info == preExpType::macroKeyword)
        return checkMacroStatementType(allExp, expPos);

      else if(allExp[expPos].info == 0)
        return 0;

      else if(expHasDescriptor(allExp, expPos))
        return checkFortranDescriptorStatementType(allExp, expPos);

      else if(allExp[expPos].info & preExpType::unknownVariable)
        return checkFortranUpdateStatementType(allExp, expPos);

      else if(allExp[expPos].info & preExpType::flowControl)
        return checkFortranFlowStatementType(allExp, expPos);

      else if(allExp[expPos].info & preExpType::specialKeyword)
        return checkFortranSpecialStatementType(allExp, expPos);

      else {
        while((expPos < allExp.leafCount) &&
              !(allExp[expPos].info & preExpType::endStatement)){

          ++expPos;
        }

        return smntType::updateStatement;
      }
    }

    int statement::checkMacroStatementType(expNode &allExp, int &expPos){
      if(expPos < allExp.leafCount){
        allExp[expPos].info = expType::macro_;
        ++expPos;
      }

      return smntType::macroStatement;
    }

    int statement::checkOccaForStatementType(expNode &allExp, int &expPos){
      if(expPos < allExp.leafCount){
        allExp[expPos].info = expType::occaFor;
        ++expPos;
      }

      return smntType::occaFor;
    }

    int statement::checkStructStatementType(expNode &allExp, int &expPos){
      if(!typeInfo::statementIsATypeInfo(allExp, expPos))
        return checkDescriptorStatementType(allExp, expPos);

      while((expPos < allExp.leafCount) &&
            !(allExp[expPos].info & preExpType::endStatement)){

        ++expPos;
      }

      return smntType::structStatement;
    }

    int statement::checkUpdateStatementType(expNode &allExp, int &expPos){
      while((expPos < allExp.leafCount) &&
            !(allExp[expPos].info & preExpType::endStatement)){

        ++expPos;
      }

      return smntType::updateStatement;
    }

    int statement::checkDescriptorStatementType(expNode &allExp, int &expPos){
      if(typeInfo::statementIsATypeInfo(allExp, expPos))
        return checkStructStatementType(allExp, expPos);

      varInfo var;
      expPos = var.loadFrom(allExp, expPos);

      if( !(var.info & varType::functionDef) ){
        while((expPos < allExp.leafCount) &&
              !(allExp[expPos].info & preExpType::endStatement)){

          ++expPos;
        }
      }

      if(var.info & varType::var)
        return smntType::declareStatement;
      else if(var.info & varType::functionDec)
        return smntType::functionPrototype;
      else
        return smntType::functionDefinition;
    }

    int statement::checkGotoStatementType(expNode &allExp, int &expPos){
      if(expPos < allExp.leafCount){
        allExp[expPos].info = expType::gotoLabel_;
        ++expPos;
      }

      return smntType::gotoStatement;
    }

    int statement::checkFlowStatementType(expNode &allExp, int &expPos){
      if(expPos < allExp.leafCount){
        allExp[expPos].info = expType::checkSInfo;

        std::string &expValue = allExp[expPos].value;
        ++expPos;

        if((expValue != "else") &&
           (expValue != "do")){

          ++expPos;
        }

        if(expValue == "for")
          return smntType::forStatement;
        else if(expValue == "while")
          return smntType::whileStatement;
        else if(expValue == "do")
          return smntType::doWhileStatement;
        else if(expValue == "if")
          return smntType::ifStatement;
        else if(expValue == "else if")
          return smntType::elseIfStatement;
        else if(expValue == "else")
          return smntType::elseStatement;
        else if(expValue == "switch")
          return smntType::switchStatement;
      }

      OCCA_CHECK(false,
                 "You found the [Waldo 2] error in:\n"
                 << allExp.toString("  "));

      return 0;
    }

    int statement::checkSpecialStatementType(expNode &allExp, int &expPos){
      if(allExp.leafCount <= expPos)
        return smntType::blankStatement;

      const bool isCaseStatement = ((allExp[expPos].value == "case") ||
                                    (allExp[expPos].value == "default"));

      while((expPos < allExp.leafCount) &&
            !(allExp[expPos].info & preExpType::endStatement)){

        ++expPos;
      }

      if(isCaseStatement)
        return smntType::caseStatement;

      return smntType::blankStatement;
    }

    int statement::checkBlockStatementType(expNode &allExp, int &expPos){
      expPos = allExp.leafCount;

      return smntType::blockStatement;
    }
    //==================================

    void statement::addType(typeInfo &type){
      scopeTypeMap[type.name] = &type;
    }

    void statement::addTypedef(const std::string &typedefName){
      typeInfo &type = *(new typeInfo);
      type.name = typedefName;
      scopeTypeMap[typedefName] = &type;
    }

    bool statement::expHasQualifier(expNode &allExp, int expPos){
      if( !(allExp[expPos].info & preExpType::qualifier) )
        return false;

      // short and long can be both:
      //    specifiers and qualifiers
      if(allExp[expPos].info == keywordType["long"]){
        if(((expPos + 1) < allExp.leafCount) &&
           (hasTypeInScope(allExp[expPos + 1].value))){

          return true;
        }
        else
          return false;
      }

      return true;
    }

    bool statement::expHasSpecifier(expNode &allExp, int expPos){
      return ((allExp[expPos].info & preExpType::specifier) ||
              ((allExp[expPos].info & preExpType::unknownVariable) &&
               ( hasTypeInScope(allExp[expPos].value) )));
    }

    bool statement::expHasDescriptor(expNode &allExp, int expPos){
      if(expHasSpecifier(allExp, expPos) ||
         expHasQualifier(allExp, expPos)){

        return true;
      }

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
    void statement::loadAllFromNode(expNode allExp, const bool parsingC){
      while(allExp.leafCount)
        loadFromNode(allExp, parsingC);
    }

    void statement::loadFromNode(expNode allExp, const bool parsingC){
      int expPos = 0;
      loadFromNode(allExp, expPos, parsingC);
    }

    void statement::loadFromNode(expNode &allExp,
                                 int &expPos,
                                 const bool parsingC){

      statement *newStatement = makeSubStatement();

      newStatement->expRoot.loadFromNode(allExp, expPos, parsingC);
      const int st = newStatement->info;

      OCCA_CHECK((st & smntType::invalidStatement) == 0,
                 "Not a valid statement");

      if(st & smntType::skipStatement){
        skipAfterStatement(allExp, expPos);

        delete newStatement;
        return;
      }

      addStatement(newStatement);

      if(st & smntType::simpleStatement){
        newStatement->loadSimpleFromNode(st,
                                         allExp,
                                         expPos,
                                         parsingC);
      }

      else if(st & smntType::flowStatement){
        if(st & smntType::forStatement)
          newStatement->loadForFromNode(st,
                                        allExp,
                                        expPos,
                                        parsingC);

        else if(st & smntType::whileStatement)
          newStatement->loadWhileFromNode(st,
                                          allExp,
                                          expPos,
                                          parsingC);

        else if(st & smntType::ifStatement)
          loadIfFromNode(st,
                         allExp,
                         expPos,
                         parsingC);

        else if(st & smntType::switchStatement)
          newStatement->loadSwitchFromNode(st,
                                           allExp,
                                           expPos,
                                           parsingC);

        else if(st & smntType::gotoStatement)
          newStatement->loadGotoFromNode(st,
                                         allExp,
                                         expPos,
                                         parsingC);
      }

      else if(st & smntType::caseStatement)
        newStatement->loadCaseFromNode(st,
                                       allExp,
                                       expPos,
                                       parsingC);

      else if(st & smntType::blockStatement)
        newStatement->loadBlockFromNode(st,
                                        allExp,
                                        expPos,
                                        parsingC);

      else if(st & smntType::functionStatement){
        if(st & smntType::functionDefinition)
          newStatement->loadFunctionDefinitionFromNode(st,
                                                       allExp,
                                                       expPos,
                                                       parsingC);

        else if(st & smntType::functionPrototype)
          newStatement->loadFunctionPrototypeFromNode(st,
                                                      allExp,
                                                      expPos,
                                                      parsingC);
      }

      else if(st & smntType::structStatement)
        newStatement->loadStructFromNode(st,
                                         allExp,
                                         expPos,
                                         parsingC);

      else if(st & smntType::blankStatement)
        newStatement->loadBlankFromNode(st,
                                        allExp,
                                        expPos,
                                        parsingC);

      else if(st & smntType::macroStatement)
        newStatement->loadMacroFromNode(st,
                                        allExp,
                                        expPos,
                                        parsingC);
    }

    expNode statement::createExpNodeFrom(const std::string &source){
      expNode ret = parserNS::splitAndLabelContent(source);
      return ret;
    }

    expNode statement::createPlainExpNodeFrom(const std::string &source){
      expNode ret = parserNS::splitAndLabelContent(source);
      ret.sInfo = this;

      ret.changeExpTypes();
      ret.initOrganization();

      return ret;
    }

    void statement::loadSimpleFromNode(const int st,
                                       expNode &allExp,
                                       int &expPos,
                                       const bool parsingC){

      if(allExp.leafCount)
        allExp.removeNode(0);
    }

    void statement::loadOneStatementFromNode(const int st,
                                             expNode &allExp,
                                             int &expPos,
                                             const bool parsingC){

      if(allExp.leafCount == 0)
        return;

      if(parsingC){
        if(allExp.info == preExpType::startBrace)
          loadAllFromNode(allExp);
        else
          loadFromNode(allExp);
      }
      else{
        loadFromNode(allExp, parsingC);
      }
    }

    void statement::loadForFromNode(const int st,
                                    expNode &allExp,
                                    int &expPos,
                                    const bool parsingC){

      if(parsingC)
        loadOneStatementFromNode(st, allExp, expPos, parsingC);
      else
        loadUntilFortranEnd(allExp, expPos);
    }

    void statement::loadWhileFromNode(const int st,
                                      expNode &allExp,
                                      int &expPos,
                                      const bool parsingC){

      if(parsingC){
        loadOneStatementFromNode(st, allExp, expPos, parsingC);

        if(st == smntType::whileStatement) {
          // Re-use the while-loop load
          info = smntType::whileStatement;

          expRoot.loadFromNode(allExp, expPos);

          info = smntType::doWhileStatement;

          // Skip the [;] after [while()]
          if((0 < allExp.leafCount) &&
             (allExp[0].value == ";")){

            allExp.removeNode(0);
          }
        }
      }
      else{
        loadUntilFortranEnd(allExp, expPos);
      }
    }

    void statement::loadIfFromNode(const int st_,
                                   expNode &allExp,
                                   int &expPos,
                                   const bool parsingC){

      statement *newStatement = statementEnd->value;

      if(parsingC){
        newStatement->loadOneStatementFromNode(st_,
                                               allExp,
                                               expPos,
                                               parsingC);

        int st      = findStatementType(allExp, expPos, parsingC);
        int stCheck = smntType::elseIfStatement;

        while(true){
          if(st != stCheck){
            if(stCheck == smntType::elseIfStatement)
              stCheck = smntType::elseStatement;
            else
              break;
          }
          else if(expPos < allExp.leafCount){
            break;
          }
          else{
            newStatement = makeSubStatement();
            newStatement->expRoot.loadFromNode(allExp, expPos, parsingC);

            OCCA_CHECK((st & smntType::invalidStatement) == 0,
                       "Not a valid statement");

            addStatement(newStatement);

            newStatement->loadOneStatementFromNode(st,
                                                   allExp,
                                                   expPos,
                                                   parsingC);

            if(expPos < allExp.leafCount)
              st = findStatementType(allExp, expPos);
          }
        }
      }
      else{
        if(allExp[expPos].value != "IF"){
          newStatement->loadUntilFortranEnd(allExp, expPos);
          return;
        }

        while(expPos < allExp.leafCount){
          if(allExp[expPos].value == "THEN"){
            newStatement->loadUntilFortranEnd(allExp, expPos);
            return;
          }
        }

        // [IF][(...)][load this]
        expPos += 2;
        newStatement->loadFromNode(allExp, expPos, parsingFortran);
      }
    }

    // [-] Missing Fortran
    void statement::loadSwitchFromNode(const int st,
                                       expNode &allExp,
                                       int &expPos,
                                       const bool parsingC){

      if(parsingC){
        loadOneStatementFromNode(st,
                                 allExp,
                                 expPos,
                                 parsingC);
      }
      else {
        loadUntilFortranEnd(allExp, expPos);
      }
    }

    // [-] Missing Fortran
    void statement::loadCaseFromNode(const int st,
                                     expNode &allExp,
                                     int &expPos,
                                     const bool parsingC){

      skipUntilFortranStatementEnd(allExp, expPos);
    }

    // [-] Missing Fortran
    void statement::loadGotoFromNode(const int st,
                                     expNode &allExp,
                                     int &expPos,
                                     const bool parsingC){

      skipUntilFortranStatementEnd(allExp, expPos);
    }

    void statement::loadFunctionDefinitionFromNode(const int st,
                                                   expNode &allExp,
                                                   int &expPos,
                                                   const bool parsingC){
      if(parsingC){
        if(expPos < allExp.leafCount)
          loadAllFromNode(allExp[expPos], parsingC);
      }
      else
        return loadUntilFortranEnd(allExp, expPos);
    }

    // [-] Missing Fortran
    void statement::loadFunctionPrototypeFromNode(const int st,
                                                  expNode &allExp,
                                                  int &expPos,
                                                  const bool parsingC){

      skipUntilFortranStatementEnd(allExp, expPos);
    }

    // [-] Missing Fortran
    void statement::loadBlockFromNode(const int st,
                                      expNode &allExp,
                                      int &expPos,
                                      const bool parsingC){

      skipUntilFortranStatementEnd(allExp, expPos);
    }

    // [-] Missing Fortran
    void statement::loadStructFromNode(const int st,
                                       expNode &allExp,
                                       int &expPos,
                                       const bool parsingC){

      skipUntilFortranStatementEnd(allExp, expPos);
    }

    // [-] Missing
    void statement::loadBlankFromNode(const int st,
                                      expNode &allExp,
                                      int &expPos,
                                      const bool parsingC){

      skipUntilFortranStatementEnd(allExp, expPos);
    }

    // [-] Missing
    void statement::loadMacroFromNode(const int st,
                                      expNode &allExp,
                                      int &expPos,
                                      const bool parsingC){

      skipUntilFortranStatementEnd(allExp, expPos);
    }

    //  ---[ Fortran ]--------
    // [+] Missing
    int statement::checkFortranStructStatementType(expNode &allExp, int &expPos){
      skipUntilFortranStatementEnd(allExp, expPos);

      return smntType::structStatement;
    }

    int statement::checkFortranUpdateStatementType(expNode &allExp, int &expPos){
      skipUntilFortranStatementEnd(allExp, expPos);

      return smntType::updateStatement;
    }

    int statement::checkFortranDescriptorStatementType(expNode &allExp, int &expPos){
      if(((expPos + 1) < allExp.leafCount)        &&
         (allExp[expPos].value     == "IMPLICIT") &&
         (allExp[expPos + 1].value == "NONE")){

        skipUntilFortranStatementEnd(allExp, expPos);

        return smntType::skipStatement;
      }

      varInfo var;
      var.loadFromFortran(allExp, expPos);

      if( !(var.info & varType::functionDef) )
        skipUntilFortranStatementEnd(allExp, expPos);

      if(var.info & varType::var)
        return smntType::declareStatement;
      else
        return smntType::functionDefinition;
    }

    int statement::checkFortranFlowStatementType(expNode &allExp, int &expPos){
      if(expPos < allExp.leafCount)
        expRoot[expPos].info = expType::checkSInfo;

      std::string &expValue = allExp[expPos].value;

      int st = 0;

      if(expValue == "DO")
        st = smntType::forStatement;
      else if(expValue == "DO WHILE")
        st = smntType::whileStatement;
      else if(expValue == "IF")
        st = smntType::ifStatement;
      else if(expValue == "ELSE IF")
        st = smntType::elseIfStatement;
      else if(expValue == "ELSE")
        st = smntType::elseStatement;
      else if(expValue == "SWITCH")
        st = smntType::switchStatement;

      // [-] Missing one-line case
      while((expPos < allExp.leafCount)     &&
            (allExp[expPos].value != "\\n") &&
            (allExp[expPos].value != ";")){

        ++expPos;
      }

      if(expPos < allExp.leafCount)
        ++expPos;

      if(st)
        return st;

      OCCA_CHECK(false,
                 "You found the [Waldo 3] error in:\n"
                 << expRoot.toString("  "));

      return 0;
    }

    int statement::checkFortranSpecialStatementType(expNode &allExp, int &expPos){
      skipUntilFortranStatementEnd(allExp, expPos);

      if(expPos < allExp.leafCount){
        if(allExp[expPos].value == "CALL"){
          return smntType::updateStatement;
        }
        else if((allExp[expPos].value == "FUNCTION") ||
                (allExp[expPos].value == "SUBROUTINE")){

          return checkFortranDescriptorStatementType(allExp, expPos);
        }
      }

      return smntType::blankStatement;
    }

    bool statement::isFortranEnd(expNode &allExp, int &expPos){
      if(allExp.leafCount <= expPos)
        return true;

      std::string expValue = allExp[expPos].value;

      if(info & smntType::functionDefinition){
        const std::string &typeName = (getFunctionVar()->baseType->name);

        if(typeName == "void")
          return (expValue == "ENDSUBROUTINE");
        else
          return (expValue == "ENDFUNCTION");
      }
      else if(info & (smntType::forStatement |
                      smntType::whileStatement)){

        return (expValue == "ENDDO");
      }
      else if(info & smntType::ifStatement){
        if(info != smntType::elseStatement){

          if((expValue == "ENDIF")   ||
             (expValue == "ELSE IF") ||
             (expValue == "ELSE")){

            return true;
          }
        }
        else
          return (expValue == "ENDIF");
      }

      return false;
    }

    void statement::loadUntilFortranEnd(expNode &allExp, int &expPos){
      while(!isFortranEnd(allExp, expPos))
        loadFromNode(allExp, expPos, parsingFortran);

      // Don't skip [ELSE IF] and [ELSE]
      if((expPos < allExp.leafCount) &&
         (allExp[expPos].value.substr(0,3) == "END")){

        skipAfterStatement(allExp, expPos);
      }
    }

    void statement::skipAfterStatement(expNode &allExp, int &expPos){
      skipUntilStatementEnd(allExp, expPos);

      if(expPos < allExp.leafCount)
        ++expPos;
    }

    void statement::skipUntilStatementEnd(expNode &allExp, int &expPos){
      while((expPos < allExp.leafCount) &&
            !(allExp[expPos].info & preExpType::endStatement)){

        ++expPos;
      }
    }

    void statement::skipUntilFortranStatementEnd(expNode &allExp, int &expPos){
      while(expPos < allExp.leafCount){
        if((allExp[expPos].value == "\\n") ||
           (allExp[expPos].value == ";")){

          break;
        }

        ++expPos;
      }
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
      loadFromNode(splitAndLabelContent(source));
    }

    void statement::addStatementsFromSource(const std::string &source){
      loadAllFromNode(splitAndLabelContent(source));
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

    bool statement::guaranteesBreak(){
      return false;
    }

    unsigned int statement::distToForLoop(){
      return distToStatementType(smntType::forStatement);
    }

    unsigned int statement::distToOccaForLoop(){
      statement *s = this;

      unsigned int dist = 0;

      while(s){
        if((s->info == smntType::occaFor) ||
           ((s->info == smntType::forStatement) &&
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
      if(info & smntType::declareStatement){
        varInfo &var = getDeclarationVarInfo(0);
        return var.hasQualifier(qualifier);
      }
      else if(info & smntType::functionStatement){
        varInfo &var = expRoot.getVarInfo(0);
        return var.hasQualifier(qualifier);
      }
      else if(info & smntType::forStatement){
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

      if(info & smntType::declareStatement){
        varInfo &var = getDeclarationVarInfo(0);
        var.addQualifier(qualifier);
      }
      else if(info & smntType::functionStatement){
        varInfo &var = expRoot.getVarInfo(0);
        var.addQualifier(qualifier, pos);
      }
      // ---[ OLD ]---
      // else if(info & smntType::forStatement){
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

      if(info & smntType::declareStatement){
        varInfo &var = getDeclarationVarInfo(0);
        var.removeQualifier(qualifier);
      }
      else if(info & smntType::functionStatement){
      }
      else if(info & smntType::forStatement){
      }
    }


    int statement::occaForInfo(){
      if(info != smntType::occaFor)
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
          //              =
          //        ->        3
          //      var  x
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
          if((nVar != &var)                || // Checking our variable update
             (n.up == NULL)                || // Update needs an assignment operator
             !isAnAssOperator(n.up->value) ||
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
      if(info & smntType::functionStatement)
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
      if(info & smntType::declareStatement)
        return expRoot.leaves[pos];

      return NULL;
    }

    std::string statement::getDeclarationVarName(const int pos){
      if(info & smntType::declareStatement){
        varInfo &var = getDeclarationVarInfo(pos);
        return var.name;
      }

      return "";
    }

    expNode* statement::getDeclarationVarInitNode(const int pos){
      if(info & smntType::declareStatement)
        return expRoot.getVariableInitNode(pos);

      return NULL;
    }

    int statement::getDeclarationVarCount(){
      if(info & smntType::declareStatement)
        return expRoot.leafCount;

      return 0;
    }

    varInfo* statement::getFunctionVar(){
      if(info & smntType::functionStatement){
        return &(expRoot.getVarInfo(0));
      }
      else if(info & smntType::updateStatement){
        statement *s = up;

        while(s &&
              !(s->info & smntType::functionStatement)){
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
      if(info & smntType::functionStatement){
        expRoot.setVarInfo(0, var);
      }
      else if(info & smntType::updateStatement){
        statement *s = up;

        while(s &&
              !(s->info & smntType::functionStatement)){
          s = s->up;
        }

        if(s)
          s->setFunctionVar(var);
      }
    }

    std::string statement::getFunctionName(){
      if(info & smntType::functionStatement){
        return getFunctionVar()->name;
      }

      OCCA_CHECK(false, "Not added yet");

      return "";
    }

    void statement::setFunctionName(const std::string &newName){
      if(info & smntType::functionStatement){
        getFunctionVar()->name = newName;
        return;
      }

      OCCA_CHECK(false, "Not added yet");
    }

    bool statement::functionHasQualifier(const std::string &qName){
      if(info & smntType::functionStatement){
        return getFunctionVar()->hasQualifier(qName);
      }

      OCCA_CHECK(false, "Not added yet");

      return false;
    }

    int statement::getFunctionArgCount(){
      if(info & smntType::functionStatement){
        return getFunctionVar()->argumentCount;
      }

      return 0;
    }

    std::string statement::getFunctionArgType(const int pos){
      if(info & smntType::functionDefinition){
        return getFunctionVar()->baseType->name;
      }

      return "";
    }

    std::string statement::getFunctionArgName(const int pos){
      if(info & smntType::functionDefinition){
        return getFunctionVar()->getArgument(pos).name;
      }

      return "";
    }

    varInfo* statement::getFunctionArgVar(const int pos){
      if(info & smntType::functionDefinition){
        return &(getFunctionVar()->getArgument(pos));
      }

      return NULL;
    }

    bool statement::hasFunctionArgVar(varInfo &var){
      if(info & smntType::functionDefinition){
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
      if( !(info & smntType::functionStatement) )
        return;

      getFunctionVar()->addArgument(pos, var);
    }

    expNode* statement::getForStatement(const int pos){
      if(info & smntType::forStatement)
        return expRoot.leaves[pos];

      return NULL;
    }

    int statement::getForStatementCount(){
      if(info & smntType::forStatement)
        return expRoot.leafCount;

      return 0;
    }
    //================================

    statement::operator std::string(){
      const std::string tab = getTab();

      statementNode *statementPos = statementStart;

      // OCCA For's
      if(info == smntType::occaFor){
        std::string ret = tab + expRoot.toString() + "{\n";

        while(statementPos){
          ret += (std::string) *(statementPos->value);
          statementPos = statementPos->right;
        }

        ret += tab + "}\n";

        return ret;
      }

      else if(info & smntType::declareStatement){
        return expRoot.toString(tab);
      }

      else if(info & (smntType::simpleStatement | smntType::gotoStatement)){
        return expRoot.toString(tab) + "\n";
      }

      else if(info & smntType::flowStatement){
        std::string ret;

        if(info != smntType::doWhileStatement){
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
           (info == smntType::doWhileStatement)){

            ret += tab + "}\n";
        }

        if(info == smntType::doWhileStatement){
          ret += ' ';
          ret += expRoot.toString();
          ret += ";\n\n";
        }

        return ret;
      }

      else if(info & smntType::caseStatement){
        return expRoot.toString(tab) + "\n";
      }

      else if(info & smntType::functionStatement){
        if(info & smntType::functionDefinition){
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
        else if(info & smntType::functionPrototype)
          return expRoot.toString(tab);
      }
      else if(info & smntType::blockStatement){
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
      else if(info & smntType::structStatement){
        return expRoot.toString(tab) + "\n";
      }
      else if(info & smntType::macroStatement){
        return tab + expRoot.value + "\n";
      }

      return expRoot.toString(tab);
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
