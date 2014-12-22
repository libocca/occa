#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
    parserBase::parserBase(){
      parsingC = true;

      macrosAreInitialized = false;
      globalScope = new statement(*this);

      warnForMissingBarriers     = true;
      warnForBarrierConditionals = true;
    }

    const std::string parserBase::parseFile(const std::string &filename,
                                            const bool parsingC_){

      parsingC = parsingC_;

      const char *cRoot = cReadFile(filename);

      const std::string parsedContent = parseSource(cRoot);

      delete [] cRoot;

      return parsedContent;
    }

    const std::string parserBase::parseSource(const char *cRoot){
      strNode *nodeRoot = splitAndPreprocessContent(cRoot);
      // nodeRoot->print();
      // throw 1;

      loadLanguageTypes();

      globalScope->loadAllFromNode(nodeRoot, parsingC);
      // std::cout << (std::string) *globalScope;
      // throw 1;

      markKernelFunctions(*globalScope);
      labelNativeKernels();

      applyToAllStatements(*globalScope, &parserBase::setupCudaVariables);
      applyToAllStatements(*globalScope, &parserBase::setupOccaVariables);

      // Broken
      // fixOccaForOrder();

      addOccaBarriers();

      addFunctionPrototypes();
      updateConstToConstant();

      addOccaFors();

      applyToAllStatements(*globalScope, &parserBase::setupOccaFors);
      // Broken
      modifyTextureVariables();

      addArgQualifiers();

      loadKernelInfos();

      // applyToAllStatements(*globalScope, &parserBase::modifyExclusiveVariables);

      return (std::string) *globalScope;
    }

    //---[ Macro Parser Functions ]-------
    std::string parserBase::getMacroName(const char *&c){
      const char *cStart = c;
      skipWord(cStart);
      skipWhitespace(cStart);
      c = cStart;
      skipWord(c);

      return std::string(cStart, c - cStart);
    }

    bool parserBase::evaluateMacroStatement(const char *&c){
      skipWhitespace(c);

      if(*c == '\0')
        return false;

      strNode *lineNode = new strNode(c);
      applyMacros(lineNode->value);
      strip(lineNode->value);

      strNode *labelNodeRoot = labelCode(lineNode);
      strNode *labelNodePos  = labelNodeRoot;

      // Check if a variable snuck in
      while(labelNodePos){
        if(labelNodePos->info & unknownVariable)
          return false;

        labelNodePos = labelNodePos->right;
      }

      typeHolder th = evaluateLabelNode(labelNodeRoot);

      return (th.doubleValue() != 0);
    }

    typeHolder parserBase::evaluateLabelNode(strNode *labelNodeRoot){
      if(labelNodeRoot->info & presetValue)
        return typeHolder(*labelNodeRoot);

      strNode *labelNodePos = labelNodeRoot;

      while(labelNodePos){
        if(labelNodePos->down){
          labelNodePos->value = evaluateLabelNode(labelNodePos->down);
          labelNodePos->info  = presetValue;
        }

        if(labelNodePos->right == NULL)
          break;

        labelNodePos = labelNodePos->right;
      }

      strNode *labelNodeEnd = labelNodePos;

      strNode *minOpNode;
      int minPrecedence, minOpType;

      labelNodePos = labelNodeRoot;

      while(true){
        minOpNode     = NULL;
        minPrecedence = 100;
        minOpType     = -1;

        while(labelNodePos){
          if(labelNodePos->info & operatorType){
            int opType = (labelNodePos->info & operatorType);

            opType &= ~qualifierType;

            if(opType & unitaryOperatorType){
              if((opType & binaryOperatorType) && // + and - operators
                 (labelNodePos->left)          &&
                 (labelNodePos->left->info & presetValue)){

                opType = binaryOperatorType;
              }
              else if((opType & rUnitaryOperatorType) &&
                      (labelNodePos->left)            &&
                      (labelNodePos->left->info & presetValue)){

                opType = rUnitaryOperatorType;
              }
              else if((opType & lUnitaryOperatorType) &&
                      (labelNodePos->right)           &&
                      (labelNodePos->right->info & presetValue)){

                opType = lUnitaryOperatorType;
              }
              else
                opType &= ~unitaryOperatorType;
            }

            const int opP = opPrecedence[opHolder(labelNodePos->value,
                                                  opType)];

            if(opP < minPrecedence){
              minOpType     = opType;
              minOpNode     = labelNodePos;
              minPrecedence = opP;
            }
          }

          labelNodePos = labelNodePos->right;
        }

        if(minOpNode == NULL){
          if(labelNodeRoot && (labelNodeRoot->right == NULL))
            return typeHolder(*labelNodeRoot);

          std::cout << "5. Error on:\n";
          labelNodeRoot->print("  ");
          throw 1;
        }
        else{
          if(minOpType & unitaryOperatorType){
            if(minOpType & lUnitaryOperatorType){
              std::string op = minOpNode->value;
              std::string a  = minOpNode->right->value;

              minOpNode->value = applyOperator(op, a);
              minOpNode->info  = presetValue;

              minOpNode->right->pop();
            }
            else if(minOpType & rUnitaryOperatorType){
              std::cout << "Postfix operator [" << *minOpNode << "] cannot be used in a macro.\n";
              throw 1;
            }
          }
          else if(minOpType & binaryOperatorType){
            minOpNode = minOpNode->left;

            std::string a  = minOpNode->value;
            std::string op = minOpNode->right->value;
            std::string b  = minOpNode->right->right->value;

            minOpNode->value = applyOperator(a, op, b);
            minOpNode->info  = presetValue;

            minOpNode->right->pop();
            minOpNode->right->pop();
          }
          else if(minOpType & ternaryOperatorType){
            minOpNode = minOpNode->left;

            std::string a  = minOpNode->value;
            std::string op = minOpNode->right->value;
            std::string b  = minOpNode->right->right->value;
            std::string c  = minOpNode->right->right->right->right->value;

            minOpNode->value = applyOperator(a, op, b, c);
            minOpNode->info  = presetValue;

            minOpNode->right->pop();
            minOpNode->right->pop();
            minOpNode->right->pop();
            minOpNode->right->pop();
          }
        }

        if(labelNodeRoot->right == NULL)
          return typeHolder(*labelNodeRoot);

        labelNodePos = labelNodeRoot;
      }

      // Shouldn't get here
      typeHolder th(labelNodeRoot->value);

      return th;
    }

    void parserBase::loadMacroInfo(macroInfo &info, const char *&c){
      const bool hasWhitespace = isWhitespace(*c);

      skipWhitespace(c);

      info.argc = 0;
      info.parts.clear();
      info.argBetweenParts.clear();

      info.parts.push_back(""); // First part

      info.isAFunction = false;

      if(*c == '\0')
        return;

      if((*c != '(') || hasWhitespace){
        const size_t chars = strlen(c);

        info.parts[0] = strip(c, chars);

        c += chars;

        return;
      }

      int partPos = 0;
      info.isAFunction = true;

      ++c; // '('

      typedef std::map<std::string,int> macroArgMap_t;
      typedef macroArgMap_t::iterator macroArgMapIterator;
      macroArgMap_t macroArgMap;

      while(*c != '\0'){
        skipWhitespace(c);
        const char *cStart = c;
        skipWord(c);

        macroArgMap[std::string(cStart, c - cStart)] = (info.argc++);

        skipWhitespace(c);

        if(*(c++) == ')')
          break;
      }

      skipWhitespace(c);

      while(*c != '\0'){
        const char *cStart = c;

        if(isAString(c)){
          skipString(c);

          info.parts[partPos] += std::string(cStart, (c - cStart));
          continue;
        }

        const int delimeterChars = skipWord(c);

        std::string word = std::string(cStart, c - cStart);

        macroArgMapIterator it = macroArgMap.find(word);

        if(it == macroArgMap.end())
          info.parts[partPos] += word;
        else{
          info.argBetweenParts.push_back(it->second);
          info.parts.push_back("");
          ++partPos;
        }

        cStart = c;
        c += delimeterChars;

        if(cStart != c)
          info.parts[partPos] += std::string(cStart, c - cStart);

        skipWhitespace(c);
      }
    }

    int parserBase::loadMacro(const std::string &line, const int state){
      const char *c = (line.c_str() + 1); // 1 = #

      while(*c != '\0'){
        skipWhitespace(c);
        const char *cEnd = c;
        skipToWhitespace(cEnd);

        if(stringsAreEqual(c, (cEnd - c), "if")){
          c = cEnd;

          bool isTrue = evaluateMacroStatement(c);

          if(isTrue)
            return (startHash | readUntilNextHash);
          else
            return (startHash | ignoreUntilNextHash);
        }

        else if(stringsAreEqual(c, (cEnd - c), "elif")){
          if((state & readUntilNextHash) || (state & ignoreUntilEnd))
            return ignoreUntilEnd;

          c = cEnd;

          bool isTrue = evaluateMacroStatement(c);

          if(isTrue)
            return readUntilNextHash;
          else
            return ignoreUntilNextHash;
        }

        else if(stringsAreEqual(c, (cEnd - c), "else")){
          if((state & readUntilNextHash) || (state & ignoreUntilEnd))
            return ignoreUntilEnd;
          else
            return readUntilNextHash;
        }

        else if(stringsAreEqual(c, (cEnd - c), "ifdef")){
          std::string name = getMacroName(c);

          if(macroMap.find(name) != macroMap.end())
            return (startHash | readUntilNextHash);
          else
            return (startHash | ignoreUntilNextHash);
        }

        else if(stringsAreEqual(c, (cEnd - c), "ifndef")){
          std::string name = getMacroName(c);

          if(macroMap.find(name) != macroMap.end())
            return (startHash | ignoreUntilNextHash);
          else
            return (startHash | readUntilNextHash);
        }

        else if(stringsAreEqual(c, (cEnd - c), "endif")){
          return doneIgnoring;
        }

        else if(stringsAreEqual(c, (cEnd - c), "define")){
          if(state & ignoring)
            return state;

          std::string name = getMacroName(c);
          int pos;

          if(macroMap.find(name) == macroMap.end()){
            pos = macros.size();
            macros.push_back( macroInfo() );
            macroMap[name] = pos;
          }
          else
            pos = macroMap[name];

          macroInfo &info = macros[pos];
          info.name = name;

          loadMacroInfo(info, c);

          return state;
        }

        else if(stringsAreEqual(c, (cEnd - c), "undef")){
          if(state & ignoring)
            return state;

          std::string name = getMacroName(c);

          if(macroMap.find(name) != macroMap.end())
            macroMap.erase(name);

          return state;
        }

        else if(stringsAreEqual(c, (cEnd - c), "include")){
          if(state & ignoring)
            return state;

          // Stick include file here

          return state;
        }

        else if(stringsAreEqual(c, (cEnd - c), "pragma"))
          return (state | keepMacro);

        else
          return (state | keepMacro);

        c = cEnd;
      }

      return state;
    }

    void parserBase::applyMacros(std::string &line){
      const char *c = line.c_str();
      std::string newLine = "";

      bool foundMacro = false;

      while(*c != '\0'){
        const char *cStart = c;

        if(isAString(c)){
          skipString(c);

          newLine += std::string(cStart, (c - cStart));
          continue;
        }

        int delimeterChars = skipWord(c);

        std::string word = std::string(cStart, c - cStart);

        macroMapIterator it = macroMap.find(word);

        if((delimeterChars == 2) &&
           stringsAreEqual(c, delimeterChars, "##") &&
           it != macroMap.end()){
          macroInfo &info = macros[it->second];
          word = info.parts[0];
        }

        while((delimeterChars == 2) &&
              stringsAreEqual(c, delimeterChars, "##")){
          c += 2;

          cStart = c;
          delimeterChars = skipWord(c);

          std::string word2 = std::string(cStart, c - cStart);

          it = macroMap.find(word2);

          if(it != macroMap.end()){
            macroInfo &info = macros[it->second];
            word += info.parts[0];
          }
          else
            word += word2;
        }

        it = macroMap.find(word);

        if(it != macroMap.end()){
          foundMacro = true;

          macroInfo &info = macros[it->second];

          if(!info.isAFunction)
            newLine += info.parts[0];

          else{
            std::vector<std::string> args;

            ++c; // '('

            while(*c != '\0'){
              skipWhitespace(c);
              cStart = c;
              skipWord(c);

              args.push_back( std::string(cStart, c - cStart) );

              skipWhitespace(c);

              if(*(c++) == ')')
                break;
            }

            newLine += info.applyArgs(args);
          }
        }
        else
          newLine += word;

        cStart = c;
        c += delimeterChars;

        if(cStart != c)
          newLine += std::string(cStart, c - cStart);

        if(isWhitespace(*c)){
          newLine += ' ';
          skipWhitespace(c);
        }
      }

      line = newLine;

      if(foundMacro)
        applyMacros(line);
    }

    strNode* parserBase::preprocessMacros(strNode *nodeRoot){
      strNode *nodePos  = nodeRoot;

      std::stack<int> statusStack;

      int currentState = doNothing;

      while(nodePos){
        std::string &line = nodePos->value;
        bool ignoreLine = false;

        if(line[0] == '#'){
          const int oldState = currentState;

          currentState = loadMacro(line, currentState);

          if(currentState & keepMacro)
            currentState &= ~keepMacro;
          // Let's keep all the macros for now
          // else
          //   ignoreLine = true;

          // Nested #if's
          if(currentState & startHash){
            currentState &= ~startHash;
            statusStack.push(oldState);
          }

          if(currentState & doneIgnoring){
            if(statusStack.size()){
              currentState = statusStack.top();
              statusStack.pop();
            }
            else
              currentState = doNothing;
          }
        }
        else{
          if(!(currentState & ignoring))
            applyMacros(line);
          else
            ignoreLine = true;
        }

        if(ignoreLine){
          if(nodeRoot == nodePos)
            nodeRoot = nodePos->right;

          popAndGoRight(nodePos);
        }
        else{
          nodePos->info = macroKeywordType;
          nodePos       = nodePos->right;
        }
      }

      return nodeRoot;
    }

    strNode* parserBase::splitAndPreprocessContent(const std::string &s){
      return splitAndPreprocessContent(s.c_str());
    }

    strNode* parserBase::splitAndPreprocessContent(const char *cRoot){
      strNode *nodeRoot;

      initKeywords(parsingC);

      nodeRoot = splitContent(cRoot, parsingC);

      initMacros(parsingC);

      nodeRoot = preprocessMacros(nodeRoot);

      nodeRoot = labelCode(nodeRoot, parsingC);

      return nodeRoot;
    }
    //====================================

    void parserBase::initMacros(const bool parsingC){
      if(!parsingC)
        initFortranMacros();

      if(macrosAreInitialized)
        return;

      macrosAreInitialized = true;

      //---[ Macros ]---------------------
      loadMacro("#define kernel occaKernel");

      loadMacro("#define barrier        occaBarrier");
      loadMacro("#define localMemFence  occaLocalMemFence");
      loadMacro("#define globalMemFence occaGlobalMemFence");

      loadMacro("#define shared   occaShared");
      loadMacro("#define restrict occaRestrict");
      loadMacro("#define volatile occaVolatile");
      loadMacro("#define aligned  occaAligned");
      loadMacro("#define const    occaConst");
      loadMacro("#define constant occaConstant");

      std::string mathFunctions[16] = {
        "sqrt", "sin"  , "asin" ,
        "sinh", "asinh", "cos"  ,
        "acos", "cosh" , "acosh",
        "tan" , "atan" , "tanh" ,
        "atanh", "exp" , "log2" ,
        "log10"
      };

      for(int i = 0; i < 16; ++i){
        std::string mf = mathFunctions[i];
        std::string cmf = mf;
        cmf[0] += ('A' - 'a');

        loadMacro("#define "       + mf  + " occa"       + cmf);
        loadMacro("#define fast"   + cmf + " occaFast"   + cmf);
        loadMacro("#define native" + cmf + " occaNative" + cmf);
      }

      //---[ CUDA Macros ]----------------
      loadMacro("#define __global__ occaKernel");

      loadMacro("#define __syncthreads()       occaBarrier(occaGlobalMemFence)");
      loadMacro("#define __threadfence_block() occaBarrier(occaLocalMemFence)");
      loadMacro("#define __threadfence()       occaBarrier(occaGlobalMemFence)");

      loadMacro("#define __shared__   occaShared");
      loadMacro("#define __restrict__ occaRestrict");
      loadMacro("#define __volatile__ occaVolatile");
      loadMacro("#define __constant__ occaConstant");

      loadMacro("#define __device__ occaFunction");

      //---[ OpenCL Macros ]--------------
      loadMacro("#define __kernel occaKernel");

      loadMacro("#define CLK_LOCAL_MEM_FENCE  occaLocalMemFence");
      loadMacro("#define CLK_GLOBAL_MEM_FENCE occaGlobalMemFence");

      loadMacro("#define __local    occaShared");
      loadMacro("#define __global   occaPointer");
      loadMacro("#define __constant occaConstant");

      loadMacro("#define get_num_groups(X)  occaOuterDim##X");
      loadMacro("#define get_group_id(X)    occaOuterId##X ");
      loadMacro("#define get_local_size(X)  occaInnerDim##X");
      loadMacro("#define get_local_id(X)    occaInnerId##X ");
      loadMacro("#define get_global_size(X) occaGlobalDim##X");
      loadMacro("#define get_global_id(X)   occaGlobalId##X ");
    }

    void parserBase::loadLanguageTypes(){
      int parts[6]            = {1, 2, 3, 4, 8, 16};
      std::string suffix[6]   = {"", "2", "3", "4", "8", "16"};
      std::string baseType[7] = {"int"  ,
                                 "bool" ,
                                 "char" ,
                                 "long" ,
                                 "short",
                                 "float",
                                 "double"};

      for(int t = 0; t < 7; ++t){
        for(int n = 0; n < 6; ++n){
          typeInfo &type = *(new typeInfo);

          type.name     = baseType[t] + suffix[n];
          type.baseType = &type;

          globalScope->scopeTypeMap[type.name] = &type;

#if 0
          if(n){
            type.addQualifier("struct");

            for(int n2 = 0; n2 < parts[n]; ++n2){
              typeInfo &uType = *(new typeInfo);
              uType.addQualifier("union");

              if(n2 < 4){
                std::string varName = "w";
                varName[0] += ((n2 + 1) % 4);

                typeInfo &sDef = uType.addType(varName);
                sType.name = baseType[t];
              }

              if(n2 < 10){
                std::string varName = "s";
                varName += '0' + n2;

                typeInfo &sDef = (n2 < 4) ? uType.addType(varName) : type.addType(varName);
                sType.name = baseType[t];
              }
              else{
                std::string varName = "s";

                typeInfo &sDef1 = uType.addType(varName + (char) ('a' + (n2 - 10)));
                sDef1.name = baseType[t];

                typeInfo &sDef2 = uType.addType(varName + (char) ('A' + (n2 - 10)));
                sDef2.name = baseType[t];
              }

              if((n2 < 4) || (10 <= n2))
                type.addType(&uType);
              else
                delete &uType;
            }
          }
#endif
        }
      }

      globalScope->addTypedef("void");
      globalScope->addTypedef("__builtin_va_list");
    }

    void parserBase::initFortranMacros(){
      if(macrosAreInitialized)
        return;

      macrosAreInitialized = true;
    }

    void parserBase::applyToAllStatements(statement &s,
                                          applyToAllStatements_t func){
      (this->*func)(s);

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        applyToAllStatements(*(statementPos->value), func);
        statementPos = statementPos->right;
      }
    }

    void parserBase::applyToStatementsUsingVar(varInfo &info,
                                               applyToStatementsUsingVar_t func){
      varUsedMapIterator it = varUsedMap.find(&info);

      if(it != varUsedMap.end()){
        statementNode *sn = it->second.right;

        while(sn){
          (this->*func)(info, *(sn->value));

          sn = sn->right;
        }
      }
    }

    bool parserBase::statementIsAKernel(statement &s){
      if(s.info & functionStatementType){
        if(s.hasQualifier("occaKernel"))
          return true;
      }

      return false;
    }

    statement* parserBase::getStatementKernel(statement &s){
      if(statementIsAKernel(s))
        return &s;

      statement *sUp = &s;

      while(sUp){
        if(statementIsAKernel(*sUp))
          return sUp;

        sUp = sUp->up;
      }

      return sUp;
    }
    statement* parserBase::getStatementOuterMostLoop(statement &s){
      statement *ret = ((s.info == occaForType) ? &s : NULL);

      statement *sUp = &s;

      while(sUp){
        if(sUp->info == occaForType)
          ret = sUp;

        sUp = sUp->up;
      }

      return ret;
    }

    bool parserBase::statementKernelUsesNativeOCCA(statement &s){
      statement *sKernel = getStatementKernel(s);

      if(sKernel == NULL)
        return false;

      std::string check = obfuscate("native", "occa");
      varInfo *info = sKernel->hasVariableInScope(check);

      if(info != NULL)
        return true;
      else
        return false;
    }

    bool parserBase::statementKernelUsesNativeOKL(statement &s){
      statement *sKernel = getStatementKernel(s);

      if(sKernel == NULL)
        return false;

      std::string check = obfuscate("native", "okl");
      varInfo *info = sKernel->hasVariableInScope(check);

      if(info != NULL)
        return true;
      else
        return false;
    }

    bool parserBase::statementKernelUsesNativeLanguage(statement &s){
      if(statementKernelUsesNativeOKL(s))
        return true;

      if(statementKernelUsesNativeOCCA(s))
        return true;

      return false;
    }

    void parserBase::addOccaForCounter(statement &s,
                                       const std::string &ioLoop,
                                       const std::string &loopNest,
                                       const std::string &loopIters){
      //---[ Add loop counter ]---------
      varInfo ioDimVar;
      ioDimVar.name = obfuscate(ioLoop);
      ioDimVar.addQualifier(loopNest);

      varInfo *ioDimVar2 = s.hasVariableInScope(ioDimVar.name);

      if(ioDimVar2 == NULL){
        statement *sPlace = getStatementOuterMostLoop(s);

        if(sPlace == NULL)
          sPlace = getStatementKernel(s);

        sPlace->addVariable(ioDimVar);
      }
      else{
        if(!ioDimVar2->hasQualifier(loopNest))
          ioDimVar2->addQualifier(loopNest);
      }

      //---[ Add loop iterations ]------
      if(loopIters.size()){
        varInfo ioIterVar;
        ioIterVar.name = obfuscate("loop", "iters");

        varInfo *ioIterVar2 = s.hasVariableInScope(ioIterVar.name);

        if(ioIterVar2 == NULL){
          statement &sOuterLoop = *(getStatementOuterMostLoop(s));
          ioIterVar2 = &(sOuterLoop.addVariable(ioIterVar));
        }

        ioIterVar2->addQualifier(loopIters);
      }
    }

    void parserBase::setupOccaFors(statement &s){
      if( !(s.info & forStatementType) ||
          (s.getForStatementCount() <= 3) ){

        return;
      }

      statement *spKernel = getStatementKernel(s);

      if(spKernel == NULL)
        return;

      if(statementKernelUsesNativeOCCA(s))
        return;

      occaLoopInfo loopInfo(s, parsingC);

      std::string ioLoopVar, ioLoop, loopNest;
      std::string iter, start;
      std::string bound, iterCheck;
      std::string opStride, opSign, iterOp;

      loopInfo.getLoopInfo(ioLoopVar, ioLoop, loopNest);

      loopInfo.getLoopNode1Info(iter, start);
      loopInfo.getLoopNode2Info(bound, iterCheck);
      loopInfo.getLoopNode3Info(opStride, opSign, iterOp);

      std::string setupExp = loopInfo.getSetupExpression();

      std::string stride = ((opSign[0] == '-') ? "-(" : "(");
      stride += opStride;
      stride += ")";
      //================================

      std::stringstream ss;

      // Working Dim
      ss << ioLoopVar << '[' << loopNest << "] = "
         << '('
         <<   "((" << bound << ") - (" << start << ") + (" << stride << " - 1))"
         <<   " / (" << stride << ")"
         << ");";

      addOccaForCounter(s, ioLoop, loopNest, ss.str());

      ss.str("");

      if(opStride != "1"){
        ss << setupExp;

        ss << ' '
           << opSign
           << " (occa" << ioLoop << "Id" << loopNest
           << " * (" << opStride << "));";
      }
      else{
        ss << setupExp;

        ss << ' '
           << opSign
           << " occa" << ioLoop << "Id" << loopNest << ";";
      }

      varInfo &iterVar = *(s.hasVariableInScope(iter));

      s.removeFromUpdateMapFor(iterVar);
      s.removeFromUsedMapFor(iterVar);

      s.scopeVarMap.erase(iter);

      s.pushSourceLeftOf(s.statementStart, ss.str());

      std::string occaForName = "occa" + ioLoop + "For" + loopNest;

      s.expRoot.info  = expType::occaFor;
      s.expRoot.value = occaForName;
      s.expRoot.free();

      s.info = occaForType;
    }

    bool parserBase::statementHasOccaOuterFor(statement &s){
      if(s.info == occaForType){
        std::string &forName = s.expRoot.value;

        if((forName.find("occaOuterFor") != std::string::npos) &&
           ((forName == "occaOuterFor0") ||
            (forName == "occaOuterFor1") ||
            (forName == "occaOuterFor2"))){

          return true;
        }
      }

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        if( statementHasOccaOuterFor(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    bool parserBase::statementHasOccaFor(statement &s){
      if(s.info == occaForType)
        return true;

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        if( statementHasOccaFor(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    bool parserBase::statementHasOklFor(statement &s){
      if(s.info == forStatementType)
        return (s.getForStatementCount() == 4);

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        if( statementHasOklFor(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    bool parserBase::statementHasOccaStuff(statement &s){
      if(statementHasOklFor(s))
        return true;

      if(statementHasOccaOuterFor(s))
        return true;

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        if( statementHasOccaStuff(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    void parserBase::markKernelFunctions(statement &s){
      statementNode *snPos = s.statementStart;

      while(snPos){
        statement &s2 = *(snPos->value);

        if( !(s2.info & functionStatementType) ||
            statementIsAKernel(s2) ){

          snPos = snPos->right;
          continue;
        }

        if(statementHasOccaStuff(s2)){
          varInfo &fVar = *(s2.getFunctionVar());
          fVar.addQualifier("occaKernel", 0);
        }

        snPos = snPos->right;
      }
    }

    void parserBase::labelNativeKernels(){
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos){
        statement &s = *(statementPos->value);

        if(statementIsAKernel(s) && // Kernel
           (s.statementStart != NULL)){

          bool hasOccaFor = statementHasOccaFor(s);
          bool hasOklFor  = statementHasOklFor(s);

          if(hasOccaFor | hasOklFor){
            varInfo nativeCheckVar;

            if(hasOccaFor)
              nativeCheckVar.name = obfuscate("native", "occa");
            else
              nativeCheckVar.name = obfuscate("native", "okl");

            varInfo *nativeCheckVar2 = s.hasVariableInScope(nativeCheckVar.name);

            if(nativeCheckVar2 == NULL)
              s.addVariable(nativeCheckVar);
          }
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::setupCudaVariables(statement &s){
      if((!(s.info & simpleStatementType)    &&
          !(s.info & forStatementType)       &&
          !(s.info & functionStatementType)) ||
         // OCCA for's don't have arguments
         (s.info == occaForType))
        return;

      if(getStatementKernel(s) == NULL)
        return;

      // [-] Go Franken-kernels ...
      // if(statementKernelUsesNativeLanguage(s))
      //   return;

      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        std::string &value = flatRoot[i].value;
        std::string occaValue;

        bool compressing = false;

        if(value == "threadIdx"){
          compressing = true;
          occaValue = "occaInnerId";
        }
        else if(value == "blockDim"){
          compressing = true;
          occaValue = "occaInnerDim";
        }
        else if(value == "blockIdx"){
          compressing = true;
          occaValue = "occaOuterId";
        }
        else if(value == "gridDim"){
          compressing = true;
          occaValue = "occaOuterDim";
        }

        if(compressing){
          expNode &leaf    = *(flatRoot[i].up);
          const char coord = (leaf[1].value[0] + ('0' - 'x'));

          leaf.info  = expType::presetValue;
          leaf.value = occaValue + coord;

          leaf.leafCount = 0;
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void parserBase::addFunctionPrototypes(){
      std::map<std::string,bool> prototypes;

      statementNode *statementPos = globalScope->statementStart;

      while(statementPos){
        statement &s = *(statementPos->value);

        if(s.info & functionPrototypeType)
          prototypes[s.getFunctionName()] = true;

        statementPos = statementPos->right;
      }

      statementPos = globalScope->statementStart;

      while(statementPos){
        statement &s = *(statementPos->value);

        if(s.info & functionStatementType){
          if(s.hasQualifier("occaKernel")){
            statementPos = statementPos->right;
            continue;
          }

          if(!s.hasQualifier("occaFunction"))
            s.addQualifier("occaFunction");

          if( !(s.info & functionDefinitionType) ){
            statementPos = statementPos->right;
            continue;
          }

          if(prototypes.find( s.getFunctionName() ) == prototypes.end()){
            globalScope->pushSourceLeftOf(statementPos,
                                          (std::string) *(s.getFunctionVar()));
          }
        }

        statementPos = statementPos->right;
      }
    }

    int parserBase::statementOccaForNest(statement &s){
      if((s.info != forStatementType) ||
         (s.getForStatementCount() != 4)){

        return notAnOccaFor;
      }

      int ret = notAnOccaFor;

      expNode &labelNode = *(s.getForStatement(3));

      const std::string &forName = (std::string) labelNode;

      if(isAnOccaOuterTag(forName)){
        ret = ((1 + forName[5] - '0') << occaOuterForShift);
      }
      else if(isAnOccaInnerTag(forName)){
        ret = ((1 + forName[5] - '0') << occaInnerForShift);
      }

      return ret;
    }

    bool parserBase::statementIsAnOccaFor(statement &s){
      const int nest = statementOccaForNest(s);

      return !(nest & notAnOccaFor);
    }

    void parserBase::addOccaBarriers(){
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos){
        statement *s = statementPos->value;

        if(statementIsAKernel(*s)      && // Kernel
           (s->statementStart != NULL)){  //   not empty

          addOccaBarriersToStatement(*s);
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::addOccaBarriersToStatement(statement &s){
      statementNode *statementPos = s.statementStart;

      statementNode *lastLoop = NULL;

      while(statementPos){
        statement &s2 = *(statementPos->value);

        int occaType = statementOccaForNest(s2);

        if((occaType == notAnOccaFor) ||
           !(occaType & occaInnerForMask)){

          addOccaBarriersToStatement(s2);
        }
        else if(lastLoop == NULL){
          lastLoop = statementPos;
        }
        else{
          statementNode *firstLoop = lastLoop;
          statementNode *snPos     = firstLoop->right;
          lastLoop = statementPos;

          while(snPos != lastLoop){
            statement &s3 = *(snPos->value);

            if(s3.hasStatementWithBarrier())
              break;

            snPos = snPos->right;
          }

          if((snPos == lastLoop) &&
             warnForMissingBarriers){

            std::cout << "Warning: Placing a local barrier between:\n"
                      << "---[ A ]--------------------------------\n"
                      << *(firstLoop->value)
                      << "---[ B ]--------------------------------\n"
                      << *(lastLoop->value)
                      << "========================================\n";

            s.pushSourceLeftOf(lastLoop, "occaBarrier(occaLocalMemFence)");
          }
        }

        statementPos = statementPos->right;
      }
    }

    bool parserBase::statementHasBarrier(statement &s){
      return s.hasBarrier();
    }

    void parserBase::fixOccaForStatementOrder(statement &origin,
                                              statementNode *sn){
      int innerLoopCount = -1;

      while(sn){
        std::vector<statement*> forLoopS;
        std::vector<int> nestStack, nestStackInv;

        statement &s = *(sn->value);

        const int sNest = statementOccaForNest(s);

        if(sNest & notAnOccaFor){
          sn = sn->right;
          continue;
        }

        const bool isAnInnerLoop = (sNest & occaInnerForMask);

        const int shift = (isAnInnerLoop ? occaInnerForShift : occaOuterForShift);
        const int mask  = (isAnInnerLoop ? occaInnerForMask  : occaOuterForMask);

        statement *sp = &s;

        forLoopS.push_back(sp);
        nestStack.push_back((sNest >> shift) - 1);

        int loopCount = 1;

        sp = sp->statementStart->value;

        while(sp){
          const int nest = statementOccaForNest(*sp);

          if(nest & (~mask))
            break;

          forLoopS.push_back(sp);
          nestStack.push_back((nest >> shift) - 1);

          ++loopCount;

          sp = sp->statementStart->value;
        }

        if(isAnInnerLoop){
          if(innerLoopCount == -1){
            innerLoopCount = loopCount;
          }
          else{
            if(loopCount != innerLoopCount){
              std::cout << "Inner loops are inconsistent in:\n"
                        << origin << '\n';

              throw 1;
            }
          }
        }
        else{
          // Re-order inner loops
          fixOccaForStatementOrder(origin, (sp->up->statementStart));
        }

        nestStackInv.resize(loopCount);

        for(int i = 0; i < loopCount; ++i)
          nestStackInv[ nestStack[i] ] = i;

        for(int i = 0; i < loopCount; ++i){
          const int sInv1 = (loopCount - i - 1);
          const int sInv2 = nestStack[i];

          if(sInv1 == sInv2)
            continue;

          const int s1 = nestStackInv[sInv1];
          const int s2 = nestStackInv[sInv2];

          swapValues(nestStack[s1]      , nestStack[s2]);
          swapValues(nestStackInv[sInv1], nestStackInv[sInv2]);

          // [-] Warning: It doesn't swap variables defined in the for-loop
          //              Should be ok ...
          expNode::swap(forLoopS[s1]->expRoot, forLoopS[s2]->expRoot);
        }

        for(int i = 0; i < loopCount; ++i){
          if(nestStack[i] != (loopCount - i - 1)){
            std::cout << "Inner loops ";

            for(int i2 = 0; i2 < loopCount; ++i2)
              std::cout << (i2 ? ", " : "[") << nestStack[i2];

            std::cout << "] have duplicates or gaps:\n"
                      << origin << '\n';

            throw 1;
          }
        }

        sn = sn->right;
      }
    }

    void parserBase::fixOccaForOrder(){
      statementNode *snPos = globalScope->statementStart;

      while(snPos){
        statement &s = *(snPos->value);

        if(statementIsAKernel(s))
          fixOccaForStatementOrder(s, s.statementStart);

        snPos = snPos->right;
      }
    }

    void parserBase::updateConstToConstant(){
      statementNode *snPos = globalScope->statementStart;

      while(snPos){
        statement &s = *(snPos->value);

        if((s.info & declareStatementType) &&
           (s.hasQualifier("occaConst"))){

          s.removeQualifier("occaConst");
          s.addQualifier("occaConstant");
        }

        snPos = snPos->right;
      }
    }

    strNode* parserBase::occaExclusiveStrNode(varInfo &var,
                                              const int depth,
                                              const int sideDepth){
      strNode *nodeRoot;

      if(var.stackPointerCount)
        nodeRoot = new strNode("occaPrivateArray");
      else
        nodeRoot = new strNode("occaPrivate");

      nodeRoot->info      = presetValue;
      nodeRoot->depth     = depth;
      nodeRoot->sideDepth = sideDepth;

      strNode *nodePos = nodeRoot->pushDown("(");

      nodePos->info  = keywordType["("];
      nodePos->depth = depth + 1;

      var.removeQualifier("exclusive");

      for(int i = 0; i < var.leftQualifiers.qualifierCount; ++i){
        nodePos       = nodePos->push(var.leftQualifiers.qualifiers[i]);
        nodePos->info = qualifierType;
      }

      nodePos       = nodePos->push(var.baseType->name);
      nodePos->info = specifierType;

      for(int i = 0; i < var.rightQualifiers.qualifierCount; ++i){
        nodePos       = nodePos->push(var.rightQualifiers.qualifiers[i]);
        nodePos->info = keywordType[var.rightQualifiers.qualifiers[i]];
      }

      nodePos       = nodePos->push(",");
      nodePos->info = keywordType[","];

      nodePos       = nodePos->push(var.name);
      nodePos->info = unknownVariable;

      if(var.stackPointerCount){
        if(1 < var.stackPointerCount){
          std::cout << "Only 1D exclusive variables are currently supported [" << var << "]\n";
          throw 1;
        }

        nodePos       = nodePos->push(",");
        nodePos->info = keywordType[","];

        nodePos       = nodePos->push((std::string) var.stackExpRoots[0]);
        nodePos->info = presetValue;
      }

      nodePos       = nodePos->push(")");
      nodePos->info = keywordType[")"];

      nodePos       = nodePos->push(";");
      nodePos->info = keywordType[";"];

      return nodeRoot;
    }

    void parserBase::addArgQualifiers(){
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos){
        statement &s = *(statementPos->value);

        if((s.info & functionDefinitionType) &&
           (s.functionHasQualifier("occaKernel"))){

          const int argc = s.getFunctionArgCount();

          for(int i = 0; i < argc; ++i){
            varInfo &argVar = *(s.getFunctionArgVar(i));

            if(argVar.pointerCount){
              if(!argVar.hasQualifier("occaPointer"))
                argVar.addQualifier("occaPointer", 0);
            }
            else{
              if(!argVar.hasRightQualifier("occaVariable"))
                argVar.addRightQualifier("occaVariable");
              if(argVar.hasRightQualifier("&"))
                argVar.removeRightQualifier("&");
            }
          }

          if(s.getFunctionArgName(0) != "occaKernelInfoArg"){
            varInfo &arg0 = *(new varInfo());

            arg0.name = "occaKernelInfoArg";

            s.addFunctionArg(0, arg0);
          }
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::modifyExclusiveVariables(statement &s){
      if( !(s.info & declareStatementType)   ||
          (getStatementKernel(s) == NULL)    ||
          (statementKernelUsesNativeOCCA(s)) ||
          (!s.hasQualifier("exclusive")) ){

        return;
      }

      std::stringstream ss;

      const int argc = s.getDeclarationVarCount();

      //---[ Setup update statement ]---
      expNode &newRoot = *(s.expRoot.clone());
      varInfo &newVar0 = newRoot.getVariableInfoNode(0)->getVarInfo();

      newVar0.leftQualifiers.clear();
      newVar0.baseType = NULL;

      bool *keepVar = new bool[argc];
      int varsKept = 0;

      for(int i = 0; i < argc; ++i){
        keepVar[i] = newRoot.variableHasInit(i);

        if(keepVar[i])
          ++varsKept;
      }

      if(varsKept){
        int pos = 0;

        for(int i = 0; i < argc; ++i){
          if(keepVar[i]){
            varInfo &newVar = newRoot.getVariableInfoNode(i)->getVarInfo();

            newVar.rightQualifiers.clear();
            newVar.removeStackPointers();

            if(pos != i)
              newRoot.leaves[pos] = newRoot.leaves[i];

            ++pos;
          }
        }

        newRoot.leafCount = varsKept;
      }
      else{
        newRoot.free();
        delete &newRoot;
      }
      //================================

      for(int i = 0; i < argc; ++i){
        varInfo &var = s.getDeclarationVarInfo(i);
        var.removeQualifier("exclusive");

        const int isPrivateArray = var.stackPointerCount;

        ss << "occaPrivate";

        if(isPrivateArray)
          ss << "Array";

        ss << "("
           << var.leftQualifiers
           << var.baseType->name
           << var.rightQualifiers << ", "
           << var.name;

        if(isPrivateArray){
          ss << ", ";

          // [-] Only supports 1D arrays
          if(1 < var.stackPointerCount){
            std::cout << "Only 1D exclusive arrays are supported:\n"
                      << "exclusive " << s << '\n';
            throw 1;
          }

          ss << var.stackExpRoots[0][0];
        }

        ss << ");";

        s.loadFromNode(labelCode( splitContent(ss.str()) ));

        s.statementEnd->value->up    = s.up;
        s.statementEnd->value->depth = s.depth;

        ss.str("");
      }

      statementNode *sn = s.getStatementNode();

      if(s.up->statementStart != sn){
        sn->left->right        = s.statementStart;
        s.statementStart->left = sn->left;
      }
      else
        s.up->statementStart = s.statementStart;

      s.statementEnd->right = sn->right;

      if(sn->right)
        sn->right->left = s.statementEnd;

      s.up->statementCount += (argc - 1);

      s.statementStart = s.statementEnd = NULL;
      s.statementCount = 0;
    }

    void parserBase::modifyTextureVariables(){
    }

    statementNode* parserBase::splitKernelStatement(statementNode *snKernel,
                                                    kernelInfo &info){
      statement &sKernel = *(snKernel->value);

      const bool usesOKL = statementKernelUsesNativeOKL(sKernel);

      statementIdMap_t idMap;
      statementVector_t sVec;
      idDepMap_t hostDepMap;

      statementVector_t loopStatements;
      intVector_t loopStatementIDs, loopOffsets, innerDims, outerDims;

      std::stringstream ss;

      sKernel.setStatementIdMap(idMap);
      sKernel.setStatementVector(idMap, sVec);

      statementNode *occaLoopRoot = getOccaLoopsInStatement(sKernel);
      statementNode *occaLoopPos  = occaLoopRoot;

      statementNode *outerLoopRoot = getOuterLoopsInStatement(sKernel);
      statementNode *outerLoopPos  = outerLoopRoot;

      const int kernelCount = length(outerLoopRoot);

      globalScope->createUniqueSequentialVariables(info.baseName,
                                                   kernelCount);

      info.nestedKernels.clear();

      // Create empty kernels
      statementNode *newSNEnd = createNestedKernelsFromLoops(snKernel,
                                                             info,
                                                             outerLoopRoot);

      // Get argument string
      std::string argsStr = getNestedKernelArgsFromLoops(sKernel);

      // Add nestedKernels argument
      setupHostKernelArgsFromLoops(sKernel);

      int kID = 0;

      // Loop through outer-loops to find statement dependencies
      //   and create the kernel
      while(occaLoopPos){
        statement &sOuter = *(occaLoopPos->value);

        varInfo *loopIter = sOuter.hasVariableInScope( obfuscate("loop", "iters") );

        loopOffsets.push_back(loopStatements.size());

        if(loopIter == NULL){
          while(occaLoopPos->value != outerLoopPos->value)
            occaLoopPos = occaLoopPos->right;

          outerLoopPos = outerLoopPos->right;

          if(outerLoopPos == NULL)
            occaLoopPos = NULL;

          continue;
        }

        qualifierInfo &loopBounds = loopIter->leftQualifiers;
        int loopPos = 0;

        const int outerDim = getOuterMostForDim(sOuter);
        const int innerDim = getInnerMostForDim(sOuter);

        outerDims.push_back(outerDim + 1);
        innerDims.push_back(innerDim + 1);

        // Break after each outer-most outer-loop (between kernels)
        // Break after each outer-most inner-loop (One work-group size (for now))
        bool firstOuter = true;
        bool firstInner = true;

        //---[ Add kernel body ]----------
        statement &ks = *(info.nestedKernels[kID]);

        // Kernel's outer-most outer-loop
        statementNode *ksnOuter = ks.statementEnd;

        // Loop through all the loops inside new kernel
        while(occaLoopPos){
          statement &s2 = *(occaLoopPos->value);

          const int forInfo = s2.occaForInfo();
          const int nest    = s2.occaForNest(forInfo);

          // Only goes through first outer and inner sets
          if(s2.isOccaInnerFor(forInfo) &&
             (nest == innerDim)){

            if(firstInner == false)
              break;

            firstInner = false;
          }
          else if(s2.isOccaOuterFor(forInfo) &&
                  (nest == outerDim)){

            if(firstOuter == false)
              break;

            firstOuter = false;
          }

          statement &ls = s2.createStatementFromSource(loopBounds[loopPos]);

          ls.addStatementDependencies(s2,
                                      idMap, sVec, hostDepMap);

          loopStatements.push_back(&ls);
          loopStatementIDs.push_back(idMap[&s2]);

          ++loopPos;
          occaLoopPos = occaLoopPos->right;
        }

        idDepMap_t depMap;

        // Get kernel dependencies
        sOuter.addNestedDependencies(idMap, sVec, depMap);

        idDepMapIterator depIt = depMap.begin();

        while(depIt != depMap.end()){
          statement &depS  = *(sVec[depIt->first]);

          if(depIt->first < idMap[&sOuter]){
            statement &depS2 = *(depS.clone());

            zeroOccaIdsFrom(depS2);

            ks.addStatement(&depS2);
          }

          ++depIt;
        }

        // Move dependencies to the front
        if(ksnOuter->right){
          statementNode *oldFirst = ks.statementStart;
          statementNode *oldEnd   = ksnOuter;

          statementNode *firstExtra = ksnOuter->right;
          statementNode *lastExtra  = ks.statementEnd;

          ks.statementStart       = firstExtra;
          ks.statementStart->left = NULL;

          ks.statementEnd        = oldEnd;
          ks.statementEnd->right = NULL;

          lastExtra->right = oldFirst;
          oldFirst->left   = lastExtra;
        }

        // Go to next outer-loop
        while(occaLoopPos){
          statement &s2 = *(occaLoopPos->value);

          const int forInfo = s2.occaForInfo();

          if(s2.isOccaOuterFor(forInfo))
            break;

          occaLoopPos = occaLoopPos->right;
        }

        ++kID;
      }

      loopOffsets.push_back(loopStatements.size());

      // Build host kernel
      const int loopCount = loopStatements.size();
      kID = 0;

      idDepMapIterator depIt = hostDepMap.begin();

      statement *blockStatement = NULL;
      statementNode newStatementStart;
      statementNode *newSNPos = &newStatementStart;

      if(loopCount == 0){
        ss << "  nestedKernels[0](" << argsStr << ");\n";

        statement &body = sKernel.createStatementFromSource(ss.str());

        newStatementStart.push(&body);

        ss.str("");
      }

      for(int loopPos = 0; loopPos < loopCount; ++loopPos){
        const int loopID = loopStatementIDs[loopPos];
        statement &ls    = *(loopStatements[loopPos]);

        while(depIt != hostDepMap.end()){
          const int sID = (depIt->first);

          if(loopID < sID)
            break;

          statement &depS = *(sVec[sID]);

          if(blockStatement)
            blockStatement->addStatement(&depS);
          else
            newSNPos = newSNPos->push(&depS);

          ++depIt;
        }

        if(loopPos == loopOffsets[kID]){
          blockStatement = new statement(ls.depth - 1,
                                         blockStatementType,
                                         &sKernel);

          newSNPos = newSNPos->push(blockStatement);

          const int outerDim = outerDims[kID];
          const int innerDim = innerDims[kID];
          const int dims     = ((outerDim < innerDim) ? innerDim : outerDim);

          ss << "const int dims = " << dims << ";\n"
             << "int outer, inner;\n";

          blockStatement->addStatementsFromSource(ss.str());

          varInfo &outerVar    = *(blockStatement->hasVariableInScope("outer"));
          statement &outerVarS = *(varUpdateMap[&outerVar].value);

          typeInfo &type = *(new typeInfo);
          type.name = "occa::dim";

          outerVarS.getDeclarationVarInfo(0).baseType = &type;

          ss.str("");

          ++kID;
        }

        blockStatement->addStatement(&ls);

        if(loopPos == (loopOffsets[kID] - 1)){
          ss << "nestedKernels[" << (kID - 1) << "].setWorkingDims(dims, inner, outer);\n"
             << "  nestedKernels[" << (kID - 1) << "](" << argsStr << ");\n";

          blockStatement->addStatementsFromSource(ss.str());

          ss.str("");

          blockStatement = NULL;
        }
      }

      sKernel.statementStart = newStatementStart.right;
      sKernel.statementEnd   = lastNode(sKernel.statementStart);

      //---[ Add kernel guards ]--------
      sKernel.up->pushSourceLeftOf(snKernel , "#if OCCA_USING_OPENMP");
      sKernel.up->pushSourceRightOf(snKernel, "#endif");

      return newSNEnd->right;
    }

    statementNode* parserBase::getOuterLoopsInStatement(statement &s){
      return getOccaLoopsInStatement(s, false);
    }

    statementNode* parserBase::getOccaLoopsInStatement(statement &s,
                                                       const bool getNestedLoops){
      statementNode *snPos = s.statementStart;

      statementNode root;
      statementNode *tail = &root;

      while(snPos){
        statement &s2 = *(snPos->value);

        if(s2.info == occaForType)
          tail = tail->push(new statementNode(&s2));

        if(getNestedLoops ||
           (s2.info != occaForType)){

          tail->right = getOccaLoopsInStatement(s2, getNestedLoops);

          if(tail->right)
            tail = lastNode(tail);
        }

        snPos = snPos->right;
      }

      return root.right;
    }

    int parserBase::kernelCountInOccaLoops(statementNode *occaLoops){
      int kernelCount = 0;

      while(occaLoops){
        ++kernelCount;

        statement &sOuter = *(occaLoops->value);

        varInfo *loopIter = sOuter.hasVariableInScope( obfuscate("loop", "iters") );

        if(loopIter == NULL){
          occaLoops = occaLoops->right;
          continue;
        }

        const int outerDim = getOuterMostForDim(sOuter) + 1;

        // Break after each outer-most outer-loop (between kernels)
        bool firstOuter = true;

        // Loop through all the loops inside new kernel
        while(occaLoops){
          statement &s2 = *(occaLoops->value);

          const int forInfo = s2.occaForInfo();
          const int nest    = s2.occaForNest(forInfo);

          if(s2.isOccaOuterFor(forInfo) &&
             (nest == (outerDim - 1))){

            if(firstOuter == false)
              break;

            firstOuter = false;
          }

          occaLoops = occaLoops->right;
        }

        // Go to next outer-loop
        while(occaLoops){
          statement &s2 = *(occaLoops->value);

          const int forInfo = s2.occaForInfo();

          if(s2.isOccaOuterFor(forInfo))
            break;

          occaLoops = occaLoops->right;
        }
      }

      return kernelCount;
    }

    void parserBase::zeroOccaIdsFrom(statement &s){
      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        if((flatRoot[i].info & expType::presetValue) &&
           isAnOccaID(flatRoot[i].value)){

          flatRoot[i].value = "0";
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    statementNode* parserBase::createNestedKernelsFromLoops(statementNode *snKernel,
                                                            kernelInfo &info,
                                                            statementNode *outerLoopRoot){
      statement &sKernel = *(snKernel->value);

      statementNode *newSNRoot, *newSNEnd;
      std::stringstream ss;

      statementNode *outerLoopPos = outerLoopRoot;

      const std::string kernelName = sKernel.getFunctionVar()->name;
      int kID = 0;

      while(outerLoopPos){
        // Create nested kernels
        statement &ks  = *(outerLoopPos->value);
        statement &ks2 = *(new statement(ks.depth - 1,
                                         varUpdateMap,
                                         varUsedMap));

        // [occaOuterFor][#]
        const int outerDim = atoi(ks.expRoot.value.c_str() + 12);

        ks2.info = sKernel.info;

        info.nestedKernels.push_back(&ks2);

        sKernel.expRoot.cloneTo(ks2.expRoot);

        varInfo &kernelVar = *(new varInfo( ks2.getFunctionVar()->clone() ));
        ks2.setFunctionVar(kernelVar);

        ss << kernelName << kID;

        kernelVar.name = ss.str();

        ss.str("");

        // Add kernel body
        ks2.addStatement(ks.clone());

        ss << "occaParallelFor" << outerDim;

        // Add the parallel-for loops
        ks2.pushSourceLeftOf(ks2.statementStart,
                             ss.str());

        ss.str("");

        // Set ks2.up and add the kernel variable
        ks2.up = sKernel.up;
        ks2.up->addVariable(&kernelVar);

        if(kID)
          newSNEnd = newSNEnd->push(new statementNode(&ks2));
        else
          newSNRoot = newSNEnd = new statementNode(&ks2);

        ++kID;
        outerLoopPos = outerLoopPos->right;
      }

      // Squeeze new kernels after original kernel
      if(sKernel.up->statementEnd == snKernel)
        sKernel.up->statementEnd = newSNEnd;

      newSNRoot->left = snKernel;
      newSNEnd->right = snKernel->right;

      if(newSNEnd->right)
        newSNEnd->right->left = newSNEnd;

      snKernel->right = newSNRoot;

      if(sKernel.up->statementEnd == snKernel)
        sKernel.up->statementEnd = newSNEnd;

      return newSNEnd;
    }

    std::string parserBase::getNestedKernelArgsFromLoops(statement &sKernel){
      // Append arguments (occaKernelInfoArg doesn't count)
      const int argc = (sKernel.getFunctionArgCount() - 1);

      std::stringstream ss;
      std::string argsStr;

      if(argc){
        ss << sKernel.getFunctionArgName(1);

        for(int i = 1; i < argc; ++i)
          ss << ", " << sKernel.getFunctionArgName(i + 1);

        argsStr = ss.str();

        ss.str("");
      }
      else
        argsStr = "";

      return argsStr;
    }

    void parserBase::setupHostKernelArgsFromLoops(statement &sKernel){
      // Add nestedKernels argument
      varInfo &arg = *(new varInfo());
      arg.loadFrom(sKernel, labelCode( splitContent("int *nestedKernels") ));

      typeInfo &type = *(new typeInfo);
      type.name = "occa::kernel";

      arg.baseType = &type;

      sKernel.addFunctionArg(1, arg);
    }

    void parserBase::loadKernelInfos(){
      statementNode *snPos = globalScope->statementStart;

      while(snPos){
        statement &s = *(snPos->value);

        if(statementIsAKernel(s)){
          //---[ Setup Info ]-----------
          kernelInfo &info = *(new kernelInfo);

          info.name     = s.getFunctionName();
          info.baseName = info.name;
          info.nestedKernels.push_back(&s);

          kernelInfoMap[info.name] = &info;
          //============================

          snPos = splitKernelStatement(snPos, info);
        }
        else
          snPos = snPos->right;
      }
    }

    void parserBase::stripOccaFromKernel(statement &s){
      // expNode &argsNode = *(s.getFunctionArgsNode());

      // argsNode.removeNode(0);

      // std::cout
      //   << "s = " << s << '\n';
    }

    std::string parserBase::occaScope(statement &s){
      statement *currentS = &s;

      while(currentS){
        if(currentS->info == occaForType)
          break;

        currentS = currentS->up;
      }

      if(currentS == NULL)
        return "";

      return (std::string) s.expRoot;
    }

    void parserBase::incrementDepth(statement &s){
      ++s.depth;
    }

    void parserBase::decrementDepth(statement &s){
      --s.depth;
    }

    statementNode* parserBase::findStatementWith(statement &s,
                                                 findStatementWith_t func){
      statementNode *ret     = new statementNode(&s);
      statementNode *retDown = NULL;

      if((this->*func)(s))
        return ret;

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        statementNode *ret2 = findStatementWith(*(statementPos->value), func);

        if(ret2 != NULL){
          if(retDown){
            retDown = retDown->push(ret2);
          }
          else{
            ret->down = ret2;
            retDown   = ret2;
          }
        }

        statementPos = statementPos->right;
      }

      if(retDown)
        return ret;

      delete ret;

      return NULL;
    }

    int parserBase::getKernelOuterDim(statement &s){
      statementNode *statementPos = s.statementStart;

      std::string outerStr = obfuscate("Outer");
      int outerDim = -1;

      varInfo *info = s.hasVariableInScope(outerStr);

      if(info != NULL){
        const int extras = info->leftQualifierCount();

        for(int i = 0; i < extras; ++i){
          const int loopNest = (info->getLeftQualifier(i)[0] - '0');

          if(outerDim < loopNest)
            outerDim = loopNest;

          // Max Dim
          if(outerDim == 2)
            return outerDim;
        }
      }

      while(statementPos){
        const int outerDim2 = getKernelOuterDim( *(statementPos->value) );

        if(outerDim < outerDim2)
          outerDim = outerDim2;

        // Max Dim
        if(outerDim == 2)
          return outerDim;

        statementPos = statementPos->right;
      }

      return outerDim;
    }

    int parserBase::getKernelInnerDim(statement &s){
      statementNode *statementPos = s.statementStart;

      std::string innerStr = obfuscate("Inner");
      int innerDim = -1;

      varInfo *info = s.hasVariableInScope(innerStr);

      if(info != NULL){
        const int extras = info->leftQualifierCount();

        for(int i = 0; i < extras; ++i){
          const int loopNest = (info->getLeftQualifier(i)[0] - '0');

          if(innerDim < loopNest)
            innerDim = loopNest;

          // Max Dim
          if(innerDim == 2)
            return innerDim;
        }
      }

      while(statementPos){
        const int innerDim2 = getKernelInnerDim( *(statementPos->value) );

        if(innerDim < innerDim2)
          innerDim = innerDim2;

        // Max Dim
        if(innerDim == 2)
          return innerDim;

        statementPos = statementPos->right;
      }

      return innerDim;
    }

    int parserBase::getOuterMostForDim(statement &s){
      return getForDim(s, "Outer");
    }

    int parserBase::getInnerMostForDim(statement &s){
      return getForDim(s, "Inner");
    }

    int parserBase::getForDim(statement &s,
                              const std::string &tag){

      std::string outerStr = obfuscate(tag);
      int outerDim = -1;

      varInfo *info = s.hasVariableInScope(outerStr);

      if(info != NULL){
        const int extras = info->leftQualifierCount();

        for(int i = 0; i < extras; ++i){
          const int loopNest = (info->getLeftQualifier(i)[0] - '0');

          if(outerDim < loopNest)
            outerDim = loopNest;

          // Max Dim
          if(outerDim == 2)
            return outerDim;
        }

        return outerDim;
      }

      std::cout << "Error, outer-most loop doesn't contain obfuscate(\"" << tag << "\"):\n"
                << s.expRoot << '\n';
      throw 1;
    }

    void parserBase::checkPathForConditionals(statementNode *path){
      if((warnForBarrierConditionals == false) ||
         (path == NULL)                        ||
         (path->value == NULL)){

        return;
      }

      if(path->value->info & ifStatementType){
        std::cout << '\n'
                  << "+---------+--------------------------------------------------+\n"
                  << "|         | Barriers inside conditional statements will only |\n"
                  << "| Warning | work properly if it's always called or never     |\n"
                  << "|         | called.                                          |\n"
                  << "+---------+--------------------------------------------------+\n"
                  << *(path->value)
                  << "+---------+--------------------------------------------------+\n\n";
      }

      if(path->down){
        statementNode *nodePos = path->down;

        while(nodePos){
          checkPathForConditionals(nodePos);
          nodePos = nodePos->right;
        }
      }
    }

    int parserBase::findLoopSections(statement &s,
                                     statementNode *path,
                                     statementIdMap_t &loopSection,
                                     int section){
      if(s.statementCount == 0)
        return section;

      statementNode *sPos  = s.statementStart;
      statementNode *sNext = NULL;

      if((path != NULL) &&
         (path->down != NULL)){

        sNext = path->down;
      }

      while(sPos){
        if(sNext &&
           (sPos->value == sNext->value)){
          // Last one is a barrier
          if(sNext->down == NULL)
            ++section;

          section = findLoopSections(*(sPos->value),
                                     sNext,
                                     loopSection,
                                     section);

          sNext = sNext->right;
        }
        else{
          loopSection[sPos->value] = section;

          findLoopSections(*(sPos->value),
                           NULL,
                           loopSection,
                           section);
        }

        sPos = sPos->right;
      }

      return section;
    }

    bool parserBase::varInTwoSegments(varInfo &var,
                                      statementIdMap_t &loopSection){
      // Don't count shared memory
      if(var.hasQualifier("occaPointer")  ||
         var.hasQualifier("occaVariable") ||
         var.hasQualifier("occaShared"))
        return false;

      varUsedMapIterator it = varUsedMap.find(&var);

      // Variable is not used
      if(it == varUsedMap.end())
        return false;

      const int segment = loopSection[varUpdateMap[&var].value];

      statementNode *pos = (it->second).right;

      while(pos){
        if(segment != loopSection[pos->value])
          return true;

        pos = pos->right;
      }

      return false;
    }

    varInfoNode* parserBase::findVarsMovingToTop(statement &s,
                                                 statementIdMap_t &loopSection){
      // Statement defines have doubles (to know how many variables
      //                                 were defined)
      //    so ... ignore duplicates
      const bool ignoringVariables = (s.info & simpleStatementType);

      scopeVarMapIterator it = s.scopeVarMap.begin();
      varInfoNode *root = NULL;
      varInfoNode *pos  = NULL;

      if(!ignoringVariables){
        while(it != s.scopeVarMap.end()){
          varInfo &var = *(it->second);

          if(varInTwoSegments(var, loopSection)){
            // No longer const
            if(var.hasQualifier("occaConst"))
              var.removeQualifier("occaConst");

            if(root == NULL){
              root = new varInfoNode(&var);
              pos  = root;
            }
            else
              pos = pos->push(&var);
          }

          ++it;
        }
      }

      statementNode *sn = s.statementStart;

      while(sn){
        varInfoNode *pos2 = findVarsMovingToTop(*(sn->value),
                                                loopSection);

        if(pos2 != NULL){
          pos->right = pos2;
          pos2->left = pos;

          pos = lastNode(pos);
        }

        sn = sn->right;
      }

      return root;
    }

    void parserBase::splitDefineForVariable(statement &origin,
                                            varInfo &var){
      int argc   = origin.getDeclarationVarCount();
      int argPos = 0;

      for(int i = 0; i < argc; ++i){
        varInfo &argVar = origin.getDeclarationVarInfo(i);

        if(&argVar == &var){
          argPos = i;
          break;
        }
      }

      if(!var.hasQualifier("exclusive"))
        var.addQualifier("exclusive");

      if(argPos){
        statement &s = origin.pushNewStatementLeft(declareStatementType);
        s.expRoot.info      = origin.expRoot.info;
        s.expRoot.leaves    = new expNode*[argPos];
        s.expRoot.leafCount = argPos;

        for(int i = 0; i < argPos; ++i){
          varInfo &argVar = origin.getDeclarationVarInfo(i);

          s.expRoot.leaves[i]         = origin.expRoot.leaves[i];
          varUpdateMap[&argVar].value = &s;
        }
      }

      if((argPos + 1) < argc){
        const int newLeafCount = (argc - (argPos + 1));

        statement &s = origin.pushNewStatementLeft(declareStatementType);
        s.expRoot.info      = origin.expRoot.info;
        s.expRoot.leaves    = new expNode*[newLeafCount];
        s.expRoot.leafCount = newLeafCount;

        for(int i = 0; i < newLeafCount; ++i){
          varInfo &argVar = origin.getDeclarationVarInfo(argPos + 1 + i);

          s.expRoot.leaves[i]         = origin.expRoot.leaves[argPos + 1 + i];
          varUpdateMap[&argVar].value = &s;
        }
      }

      origin.expRoot.leaves[0] = origin.expRoot.leaves[argPos];
      origin.expRoot.leafCount = 1;

      if(origin.expRoot.variableHasInit(0)){
        std::stringstream ss;

        ss << var.name << " = " << *(origin.expRoot.getVariableInitNode(0)) << ";";

        origin.up->pushSourceRightOf(origin.getStatementNode(),
                                     ss.str());
      }

      return;
    }

    void parserBase::addInnerForsToStatement(statement &s,
                                             const int innerDim){
      statementNode *sn = s.statementStart;

      while(sn &&
            (sn->value->hasDescriptorVariable("occaShared") ||
             sn->value->hasDescriptorVariable("exclusive")  ||
             sn->value->hasBarrier()))
        sn = sn->right;

      while(sn){
        sn = addInnerForsBetweenBarriers(s, sn, innerDim);

        while(sn &&
              sn->value->hasBarrier()){

          sn = sn->right;
        }
      }
    }

    statementNode* parserBase::addInnerForsBetweenBarriers(statement &origin,
                                                           statementNode *includeStart,
                                                           const int innerDim){
      if(includeStart == NULL)
        return NULL;

      const int occaForType = keywordType["occaInnerFor0"];

      statement *outerMostLoop = NULL;
      statement *innerMostLoop = NULL;

      statementNode *includeEnd = includeStart;
      statementNode *returnNode;

      bool stoppedAtFor = false;

      while(includeEnd){
        if(includeEnd->value->hasBarrier())
          break;

        // [-] Re-add if we want to split between for-loops
        if(includeEnd->value->info & forStatementType){
          stoppedAtFor = true;
          break;
        }

        includeEnd = includeEnd->right;
      }

      returnNode = (includeEnd ? includeEnd->right : NULL);

      if(includeEnd && stoppedAtFor){
        statementNode *forStart = includeEnd;
        statementNode *forNode  = includeEnd;

        while(forStart->left &&
              forStart->left->value->info & macroStatementType)
          forStart = forStart->left;

        includeEnd = forStart;

        addInnerForsToStatement(*(forNode->value), innerDim);

        if(includeStart == forStart)
          return returnNode;
      }

      if(includeEnd)
        includeEnd = includeEnd->left;

      for(int i = innerDim; 0 <= i; --i){
        statement *newStatement = new statement(includeStart->value->depth,
                                                occaForType, &origin);

        newStatement->expRoot.info   = expType::occaFor;
        newStatement->expRoot.value  = "occaInnerFor";
        newStatement->expRoot.value += ('0' + i);

        if(i == innerDim){
          outerMostLoop = newStatement;
          innerMostLoop = outerMostLoop;
        }
        else{
          innerMostLoop->addStatement(newStatement);
          innerMostLoop->statementStart->value->up = innerMostLoop;
          innerMostLoop = innerMostLoop->statementStart->value;
        }
      }

      // Keep privates and shared outside inner loops
      while(includeStart &&
            (includeStart->value->hasDescriptorVariable("occaShared") ||
             includeStart->value->hasDescriptorVariable("exclusive"))){

        includeStart->value->up = innerMostLoop;
        includeStart = includeStart->right;
      }

      // Put the loop node on the origin's statements
      //   and remove nodes that are going to be put
      //   in the inner loops
      statementNode *outerMostLoopSN = new statementNode(outerMostLoop);

      if(origin.statementStart == includeStart)
        origin.statementStart = outerMostLoopSN;

      outerMostLoopSN->left = includeStart->left;

      if(includeStart->left)
        includeStart->left->right = outerMostLoopSN;

      includeStart->left = NULL;

      if(includeEnd){
        outerMostLoopSN->right = includeEnd->right;

        if(includeEnd->right)
          includeEnd->right->left = outerMostLoopSN;

        includeEnd->right = NULL;
      }

      innerMostLoop->statementStart = includeStart;
      innerMostLoop->statementEnd   = includeEnd;

      while(includeStart){
        includeStart->value->up = innerMostLoop;
        includeStart = includeStart->right;
      }

      // Increment the depth of statements in the loops
      for(int i = 0; i < innerDim; ++i){
        outerMostLoop = outerMostLoop->statementStart->value;
        applyToAllStatements(*outerMostLoop, &parserBase::incrementDepth);
      }

      applyToAllStatements(*outerMostLoop, &parserBase::incrementDepth);
      --(outerMostLoop->depth);

      return returnNode;
    }

    void parserBase::addInnerFors(statement &s){
      int innerDim = getKernelInnerDim(s);

      if(innerDim == -1){
        std::cout << "OCCA Inner for-loop count could not be calculated\n";
        throw 1;
      }

      // Get path and ignore kernel
      statementNode *sPath = findStatementWith(s, &parserBase::statementHasBarrier);

      checkPathForConditionals(sPath);

      statementIdMap_t loopSection;
      findLoopSections(s, sPath, loopSection);

      // Get private and shared vars
      varInfoNode *varRoot = findVarsMovingToTop(s, loopSection);
      varInfoNode *varPos  = lastNode(varRoot);

      statementNode *newStatementStart = NULL;
      statementNode *newStatementPos   = NULL;

      while(varPos){
        statement &origin = *(varUpdateMap[varPos->value].value);

        // Ignore kernel arguments
        if(origin.info & functionStatementType){
          varPos = varPos->left;
          continue;
        }

        varInfo &info = *(varPos->value);

        splitDefineForVariable(origin, info);

        varPos = varPos->left;
      }

      floatSharedVarsInKernel(s);
      addInnerForsToStatement(s, innerDim);
    }

    void parserBase::addOuterFors(statement &s){
      int outerDim = getKernelOuterDim(s);

      if(outerDim == -1){
        std::cout << "OCCA Outer for-loop count could not be calculated\n";
        throw 1;
      }

      statement *sPos = &s;

      for(int o = outerDim; 0 <= o; --o){
        statement *newStatement = new statement(sPos->depth + 1,
                                                occaForType, &s);

        newStatement->expRoot.info   = expType::printValue;
        newStatement->expRoot.value  = "occaOuterFor";
        newStatement->expRoot.value += ('0' + o);

        newStatement->scopeVarMap = sPos->scopeVarMap;

        statementNode *sn = sPos->statementStart;

        while(sn){
          newStatement->addStatement(sn->value);

          sn->value->up = newStatement;
          applyToAllStatements(*(sn->value), &parserBase::incrementDepth);

          statementNode *sn2 = sn->right;
          delete sn;
          sn = sn2;
        }

        sPos->statementCount = 0;
        sPos->statementStart = sPos->statementEnd = NULL;
        sPos->scopeVarMap.clear();

        sPos->addStatement(newStatement);

        sPos = newStatement;
      }
    }

    void parserBase::removeUnnecessaryBlocksInKernel(statement &s){
      statement *sPos = &s;

      // Get rid of empty blocks
      //  kernel void blah(){{  -->  kernel void blah(){
      //  }}                    -->  }
      while(sPos->statementCount == 1){
        statement *sDown = sPos->statementStart->value;

        if(sDown->info == blockStatementType){
          sPos->scopeVarMap.insert(sDown->scopeVarMap.begin(),
                                   sDown->scopeVarMap.end());

          sPos->statementCount = 0;
          sPos->statementStart = sPos->statementEnd = NULL;

          statementNode *sn = sDown->statementStart;

          while(sn){
            sPos->addStatement(sn->value);

            sn->value->up = sPos;
            applyToAllStatements(*(sn->value), &parserBase::decrementDepth);

            statementNode *sn2 = sn->right;
            delete sn;
            sn = sn2;
          }
        }
        else
          break;
      }
    }

    void parserBase::floatSharedVarsInKernel(statement &s){
      statementNode *sn = s.statementStart;

      statementNode *sharedStart = NULL;
      statementNode *sharedPos   = NULL;

      while(sn){
        statementNode *sn2 = sn;
        statement &s2      = *(sn->value);

        sn = sn->right;

        if((s2.info & declareStatementType) &&
           (s2.hasDescriptorVariable("occaShared") ||
            s2.hasDescriptorVariable("exclusive"))){

          if(s.statementStart == sn2)
            s.statementStart = sn;

          if(sn2->left)
            sn2->left->right = sn2->right;
          if(sn2->right)
            sn2->right->left = sn2->left;

          sn2->left  = NULL;
          sn2->right = NULL;

          if(sharedStart){
            sharedPos->right = sn2;
            sn2->left        = sharedPos;
            sharedPos        = sn2;
          }
          else{
            sharedStart = sn2;
            sharedPos   = sharedStart;
          }
        }
      }

      if(sharedStart){
        if(s.statementStart == NULL)
          s.statementStart = sharedStart;
        else{
          statementNode *oldStart = s.statementStart;

          s.statementStart = sharedStart;
          sharedPos->right = oldStart;
          oldStart->left   = sharedPos;
        }
      }
    }

    void parserBase::addOccaForsToKernel(statement &s){
      addInnerFors(s);
      addOuterFors(s);
    }

    void parserBase::addOccaFors(){
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos){
        statement *s = statementPos->value;

        if(statementIsAKernel(*s)            && // Kernel
           (s->statementStart != NULL)       && //   not empty
           !statementKernelUsesNativeOKL(*s) && //   not OKL
           !statementKernelUsesNativeOCCA(*s)){ //   not OCCA

          removeUnnecessaryBlocksInKernel(*s);
          addOccaForsToKernel(*s);
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::setupOccaVariables(statement &s){
      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        std::string &value = flatRoot[i].value;
        std::string occaValue;

        const bool isInnerId   = isAnOccaInnerID(value);
        const bool isOuterId   = isAnOccaOuterID(value);
        const bool isGlobalId  = isAnOccaGlobalID(value);
        const bool isInnerDim  = isAnOccaInnerDim(value);
        const bool isOuterDim  = isAnOccaOuterDim(value);
        const bool isGlobalDim = isAnOccaGlobalDim(value);

        const bool isId  = (isInnerId  || isOuterId  || isGlobalId);
        const bool isDim = (isInnerDim || isOuterDim || isGlobalDim);

        if(isId || isDim){
          std::string ioLoop, loopNest;

          if(isId){
            if(isInnerId || isOuterId){
              // [occa][-----][Id#]
              ioLoop = value.substr(4,5);
              // [occa][-----Id][#]
              loopNest = value.substr(11,1);

              addOccaForCounter(s, ioLoop, loopNest);
            }
            else { // isGlobalId
              // [occa][------Id][#]
              loopNest = value.substr(12,1);

              addOccaForCounter(s, "Inner", loopNest);
              addOccaForCounter(s, "Outer", loopNest);
            }
          }
          else { // isDim
            if(isInnerDim || isOuterDim){
              // [occa][-----][Dim#]
              ioLoop = value.substr(4,5);
              // [occa][-----Dim][#]
              loopNest = value.substr(12,1);

              addOccaForCounter(s, ioLoop, loopNest);
            }
            else { // isGlobalDim
              // [occa][------Dim][#]
              loopNest = value.substr(13,1);

              addOccaForCounter(s, "Inner", loopNest);
              addOccaForCounter(s, "Outer", loopNest);
            }
          }

        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    bool isAnOccaID(const std::string &s){
      return (isAnOccaInnerID(s) ||
              isAnOccaOuterID(s) ||
              isAnOccaGlobalID(s));
    }

    bool isAnOccaDim(const std::string &s){
      return (isAnOccaInnerDim(s) ||
              isAnOccaOuterDim(s) ||
              isAnOccaGlobalDim(s));
    }

    bool isAnOccaInnerID(const std::string &s){
      return ((s == "occaInnerId0") ||
              (s == "occaInnerId1") ||
              (s == "occaInnerId2"));
    }

    bool isAnOccaOuterID(const std::string &s){
      return ((s == "occaOuterId0") ||
              (s == "occaOuterId1") ||
              (s == "occaOuterId2"));
    }

    bool isAnOccaGlobalID(const std::string &s){
      return ((s == "occaGlobalId0") ||
              (s == "occaGlobalId1") ||
              (s == "occaGlobalId2"));
    }

    bool isAnOccaInnerDim(const std::string &s){
      return ((s == "occaInnerDim0") ||
              (s == "occaInnerDim1") ||
              (s == "occaInnerDim2"));
    }

    bool isAnOccaOuterDim(const std::string &s){
      return ((s == "occaOuterDim0") ||
              (s == "occaOuterDim1") ||
              (s == "occaOuterDim2"));
    }

    bool isAnOccaGlobalDim(const std::string &s){
      return ((s == "occaGlobalDim0") ||
              (s == "occaGlobalDim1") ||
              (s == "occaGlobalDim2"));
    }

    //==============================================

    strNode* splitContent(const std::string &str, const bool parsingC){
      return splitContent(str.c_str(), parsingC);
    }

    strNode* splitContent(const char *cRoot, const bool parsingC){
      initKeywords(parsingC);

      const char *c = cRoot;

      strNode *nodeRoot = new strNode();
      strNode *nodePos  = nodeRoot;

      int status = readingCode;

      int lineCount = 0;

      while(*c != '\0'){
        const char *cEnd = readLine(c, parsingC);

        std::string line = strip(c, cEnd - c);

        if(line.size()){
          if(!parsingC &&
             (*c == 'c')){
            c = cEnd;
            continue;
          }

          if(status != insideCommentBlock){
            status = stripComments(line, parsingC);

            strip(line);

            if(line.size()){
              nodePos->originalLine = lineCount;
              nodePos = nodePos->push(line);
            }
          }
          else{
            status = stripComments(line, parsingC);

            strip(line);

            if((status == finishedCommentBlock) && line.size()){
              nodePos->originalLine = lineCount;
              nodePos = nodePos->push(line);
            }
          }
        }

        c = cEnd;
        ++lineCount;
      }

      popAndGoRight(nodeRoot);

      return nodeRoot;
    }

    strNode* labelCode(strNode *lineNodeRoot, const bool parsingC){
      initKeywords(parsingC);

      strNode *nodeRoot = new strNode();
      strNode *nodePos  = nodeRoot;

      strNode *lineNodePos = lineNodeRoot;

      int depth = 0;
      bool firstSectionNode = false;

      while(lineNodePos){
        const std::string &line = lineNodePos->value;
        const char *cLeft = line.c_str();

        while(*cLeft != '\0'){
          skipWhitespace(cLeft);

          const char *cRight = cLeft;

          bool loadString = isAString(cLeft);
          bool loadNumber = isANumber(cLeft);

          // Case: n +1
          if(loadNumber){
            const int delimeterChars = isAWordDelimeter(cLeft, parsingC);

            if((delimeterChars == 1) &&
               ((cLeft[0] == '+') || (cLeft[0] == '-'))){

              if(nodePos->left)
                loadNumber = false;
            }
          }

          if(loadString){
            skipString(cRight);

            if(!firstSectionNode){
              nodePos = nodePos->push( std::string(cLeft, (cRight - cLeft)) );
            }
            else{
              nodePos = nodePos->pushDown( std::string(cLeft, (cRight - cLeft)) );
              firstSectionNode = false;
            }

            nodePos->info  = presetValue;
            nodePos->depth = depth;

            cLeft = cRight;
          }
          else if(loadNumber){
            const int delimeterChars = isAWordDelimeter(cLeft);

            skipNumber(cRight, parsingC);

            if(!firstSectionNode){
              nodePos = nodePos->push( std::string(cLeft, (cRight - cLeft)) );
            }
            else{
              nodePos = nodePos->pushDown( std::string(cLeft, (cRight - cLeft)) );
              firstSectionNode = false;
            }

            nodePos->info  = presetValue;
            nodePos->depth = depth;

            cLeft = cRight;
          }
          else{
            const int delimeterChars = isAWordDelimeter(cLeft, parsingC);

            if(delimeterChars){
              strNode *newNode;

              if(!parsingC){
                // Translate Fortran keywords
                std::string op(cLeft, delimeterChars);
                std::string upOp = upString(op);

                if(upOp[0] == '.'){
                  if     (upOp == ".TRUE.")  upOp = "true";
                  else if(upOp == ".FALSE.") upOp = "false";

                  else if(upOp == ".LT.")    upOp = "<";
                  else if(upOp == ".GT.")    upOp = ">";
                  else if(upOp == ".LE.")    upOp = "<=";
                  else if(upOp == ".GE.")    upOp = ">=";
                  else if(upOp == ".EQ.")    upOp = "==";

                  else if(upOp == ".NOT.")   upOp = "!";
                  else if(upOp == ".AND.")   upOp = "&&";
                  else if(upOp == ".OR.")    upOp = "||";
                  else if(upOp == ".EQV.")   upOp = "==";
                  else if(upOp == ".NEQV.")  upOp = "!=";

                  newNode = new strNode(upOp);
                }
                else if(upOp == "/="){
                  newNode = new strNode("!=");
                }
                else {
                  newNode = new strNode(op);
                }
              }
              else {
                newNode = new strNode(std::string(cLeft, delimeterChars));
              }

              newNode->info  = keywordType[newNode->value];
              newNode->depth = depth;

              if(newNode->info & startSection){
                if(!firstSectionNode)
                  nodePos = nodePos->push(newNode);
                else
                  nodePos = nodePos->pushDown(newNode);

                ++depth;

                firstSectionNode = true;
              }
              else if(newNode->info & endSection){
                if(!firstSectionNode)
                  nodePos = nodePos->up;

                delete newNode;

                --depth;

                firstSectionNode = false;
              }
              else if(newNode->info & macroKeywordType){
                newNode->value = line;

                if(!firstSectionNode)
                  nodePos = nodePos->push(newNode);
                else{
                  nodePos = nodePos->pushDown(newNode);
                  firstSectionNode = false;
                }

                cLeft = line.c_str() + strlen(line.c_str()) - delimeterChars;
              }
              else{
                if(!firstSectionNode)
                  nodePos = nodePos->push(newNode);
                else{
                  nodePos = nodePos->pushDown(newNode);
                  firstSectionNode = false;
                }
              }

              cLeft += delimeterChars;
            }
            else{
              skipWord(cRight, parsingC);

              std::string nodeValue(cLeft, (cRight - cLeft));
              keywordTypeMapIterator it;

              if(!parsingC){
                std::string upNodeValue = upString(nodeValue);

                it = keywordType.find(upNodeValue);

                if(it != keywordType.end())
                  nodeValue = upNodeValue;
              }
              else{
                it = keywordType.find(nodeValue);
              }

              if(!firstSectionNode){
                nodePos = nodePos->push(nodeValue);
              }
              else{
                nodePos = nodePos->pushDown(nodeValue);
                firstSectionNode = false;
              }

              if(it == keywordType.end())
                nodePos->info = unknownVariable;
              else{
                nodePos->info = it->second;

                if(parsingC){
                  if(checkWithLeft(nodePos, "else", "if")){
                    mergeNodeWithLeft(nodePos);
                  }
                  else if((nodePos->info & specialKeywordType) &&
                          (nodePos->value == "__attribute__")){

                    skipWhitespace(cRight);
                    skipPair(cRight);

                    // [-] Early fix
                    if(nodePos->left){
                      nodePos = nodePos->left;

                      delete nodePos->right;
                      nodePos->right = NULL;
                    }
                    else if(nodePos->up){
                      nodePos = nodePos->up;

                      delete nodePos->down;
                      nodePos->down = NULL;
                    }
                  }
                }
                else{
                  if(checkWithLeft(nodePos, "else", "if"   , parsingC) ||
                     checkWithLeft(nodePos, "do"  , "while", parsingC)){

                    mergeNodeWithLeft(nodePos, true, parsingC);
                  }
                  else if(checkWithLeft(nodePos, "end" , "do"        , parsingC) ||
                          checkWithLeft(nodePos, "end" , "if"        , parsingC) ||
                          checkWithLeft(nodePos, "end" , "function"  , parsingC) ||
                          checkWithLeft(nodePos, "end" , "subroutine", parsingC)){

                    mergeNodeWithLeft(nodePos, false, parsingC);
                  }
                }
              }

              nodePos->depth = depth;

              cLeft = cRight;
            }
          }
        }

        if(!parsingC){
          nodePos       = nodePos->push("\\n");
          nodePos->info = endStatement;
        }

        lineNodePos = lineNodePos->right;
      }

      if(nodePos != nodeRoot)
        popAndGoRight(nodeRoot);

      free(lineNodeRoot);

      return nodeRoot;
    }

    bool checkWithLeft(strNode *nodePos,
                       const std::string &leftValue,
                       const std::string &rightValue,
                       const bool parsingC){

      if(parsingC){
        return ((nodePos->left)                      &&
                (nodePos->value       == rightValue) &&
                (nodePos->left->value == leftValue));
      }

      return ((nodePos->left)                                 &&
              upStringCheck(nodePos->value      , rightValue) &&
              upStringCheck(nodePos->left->value, leftValue));
    }

    void mergeNodeWithLeft(strNode *&nodePos,
                           const bool addSpace,
                           const bool parsingC){

      if((nodePos->left) == NULL)
        return;

      strNode *leftNode = nodePos->left;

      if(addSpace){
        leftNode->value += " ";
        leftNode->value += (nodePos->value);
        nodePos->value   = (leftNode->value);
      }
      else{
        nodePos->value = ((leftNode->value) + (nodePos->value));
      }

      if(!parsingC)
        nodePos->value = upString(nodePos->value);

      nodePos->left = nodePos->left->left;

      if(nodePos->left)
        nodePos->left->right = nodePos;

      delete leftNode->pop();
    }

    void initKeywords(const bool parsingC){
      if(!parsingC){
        initFortranKeywords();
        return;
      }

      if(cKeywordsAreInitialized){
        keywordType = cKeywordType;

        return;
      }

      cKeywordsAreInitialized = true;

      //---[ Operator Info ]--------------
      cKeywordType["!"]  = lUnitaryOperatorType;
      cKeywordType["%"]  = binaryOperatorType;
      cKeywordType["&"]  = (binaryOperatorType | qualifierType);
      cKeywordType["("]  = startParentheses;
      cKeywordType[")"]  = endParentheses;
      cKeywordType["*"]  = (binaryOperatorType | qualifierType);
      cKeywordType["+"]  = (lUnitaryOperatorType | binaryOperatorType);
      cKeywordType[","]  = binaryOperatorType;
      cKeywordType["-"]  = (lUnitaryOperatorType | binaryOperatorType);
      cKeywordType["."]  = binaryOperatorType;
      cKeywordType["/"]  = binaryOperatorType;
      cKeywordType[":"]  = endStatement;
      cKeywordType[";"]  = endStatement;
      cKeywordType["<"]  = binaryOperatorType;
      cKeywordType["="]  = binaryOperatorType;
      cKeywordType[">"]  = binaryOperatorType;
      cKeywordType["?"]  = ternaryOperatorType;
      cKeywordType["["]  = startBracket;
      cKeywordType["]"]  = endBracket;
      cKeywordType["^"]  = (binaryOperatorType | qualifierType);
      cKeywordType["{"]  = startBrace;
      cKeywordType["|"]  = binaryOperatorType;
      cKeywordType["}"]  = endBrace;
      cKeywordType["~"]  = lUnitaryOperatorType;
      cKeywordType["!="] = assOperatorType;
      cKeywordType["%="] = assOperatorType;
      cKeywordType["&&"] = binaryOperatorType;
      cKeywordType["&="] = assOperatorType;
      cKeywordType["*="] = assOperatorType;
      cKeywordType["+="] = assOperatorType;
      cKeywordType["++"] = unitaryOperatorType;
      cKeywordType["-="] = assOperatorType;
      cKeywordType["--"] = unitaryOperatorType;
      cKeywordType["->"] = binaryOperatorType;
      cKeywordType["/="] = assOperatorType;
      cKeywordType["::"] = binaryOperatorType;
      cKeywordType["<<"] = binaryOperatorType;
      cKeywordType["<="] = binaryOperatorType;
      cKeywordType["=="] = binaryOperatorType;
      cKeywordType[">="] = binaryOperatorType;
      cKeywordType[">>"] = binaryOperatorType;
      cKeywordType["^="] = assOperatorType;
      cKeywordType["|="] = assOperatorType;
      cKeywordType["||"] = binaryOperatorType;

      cKeywordType["#"] = macroKeywordType;

      //---[ Types & Specifiers ]---------
      std::string suffix[6] = {"", "2", "3", "4", "8", "16"};

      for(int i = 0; i < 6; ++i){
        cKeywordType[std::string("int")    + suffix[i]] = specifierType;
        cKeywordType[std::string("bool")   + suffix[i]] = specifierType;
        cKeywordType[std::string("char")   + suffix[i]] = specifierType;
        cKeywordType[std::string("long")   + suffix[i]] = specifierType;
        cKeywordType[std::string("short")  + suffix[i]] = specifierType;
        cKeywordType[std::string("float")  + suffix[i]] = specifierType;
        cKeywordType[std::string("double") + suffix[i]] = specifierType;
      }

      cKeywordType["void"]          = specifierType;
      cKeywordType["__attribute__"] = specifierType; // [--]

      cKeywordType["long"]     = (qualifierType | specifierType);
      cKeywordType["short"]    = (qualifierType | specifierType);
      cKeywordType["signed"]   = qualifierType;
      cKeywordType["unsigned"] = qualifierType;

      cKeywordType["inline"] = qualifierType;
      cKeywordType["static"] = qualifierType;
      cKeywordType["extern"] = qualifierType;

      cKeywordType["const"]    = (qualifierType | occaKeywordType);
      cKeywordType["restrict"] = (qualifierType | occaKeywordType);
      cKeywordType["volatile"] = (qualifierType | occaKeywordType);
      cKeywordType["aligned"]  = (qualifierType | occaKeywordType);
      cKeywordType["register"] = qualifierType;

      cKeywordType["occaConst"]    = (qualifierType | occaKeywordType);
      cKeywordType["occaRestrict"] = (qualifierType | occaKeywordType);
      cKeywordType["occaVolatile"] = (qualifierType | occaKeywordType);
      cKeywordType["occaAligned"]  = (qualifierType | occaKeywordType);
      cKeywordType["occaConstant"] = (qualifierType | occaKeywordType);

      cKeywordType["class"]   = (structType);
      cKeywordType["enum"]    = (structType | qualifierType);
      cKeywordType["union"]   = (structType | qualifierType);
      cKeywordType["struct"]  = (structType | qualifierType);
      cKeywordType["typedef"] = (typedefType | qualifierType);

      //---[ Non-standard ]-------------
      cKeywordType["__attribute__"] = (qualifierType | specialKeywordType);

      //---[ C++ ]----------------------
      cKeywordType["virtual"]   = qualifierType;

      cKeywordType["namespace"] = (specifierType | structType);

      //---[ Constants ]------------------
      cKeywordType["..."]   = presetValue;
      cKeywordType["true"]  = presetValue;
      cKeywordType["false"] = presetValue;

      //---[ Flow Control ]---------------
      cKeywordType["if"]   = flowControlType;
      cKeywordType["else"] = flowControlType;

      cKeywordType["for"] = flowControlType;

      cKeywordType["do"]    = flowControlType;
      cKeywordType["while"] = flowControlType;

      cKeywordType["switch"]  = flowControlType;
      cKeywordType["case"]    = specialKeywordType;
      cKeywordType["default"] = specialKeywordType;

      cKeywordType["break"]    = specialKeywordType;
      cKeywordType["continue"] = specialKeywordType;
      cKeywordType["return"]   = specialKeywordType;
      cKeywordType["goto"]     = specialKeywordType;

      //---[ OCCA Keywords ]--------------
      cKeywordType["kernel"]    = (qualifierType | occaKeywordType);
      cKeywordType["texture"]   = (qualifierType | occaKeywordType);
      cKeywordType["shared"]    = (qualifierType | occaKeywordType);
      cKeywordType["exclusive"] = (qualifierType | occaKeywordType);

      cKeywordType["occaKernel"]   = (qualifierType | occaKeywordType);
      cKeywordType["occaFunction"] = (qualifierType | occaKeywordType);
      cKeywordType["occaDeviceFunction"] = (qualifierType | occaKeywordType);
      cKeywordType["occaPointer"]  = (qualifierType | occaKeywordType);
      cKeywordType["occaVariable"] = (qualifierType | occaKeywordType);
      cKeywordType["occaShared"]   = (qualifierType | occaKeywordType);

      cKeywordType["occaKernelInfoArg"] = (presetValue | occaKeywordType);
      cKeywordType["occaKernelInfo"]    = (presetValue | occaKeywordType);

      cKeywordType["occaPrivate"]      = (presetValue | occaKeywordType);
      cKeywordType["occaPrivateArray"] = (presetValue | occaKeywordType);

      cKeywordType["barrier"]        = (presetValue | occaKeywordType);
      cKeywordType["localMemFence"]  = (presetValue | occaKeywordType);
      cKeywordType["globalMemFence"] = (presetValue | occaKeywordType);

      cKeywordType["occaBarrier"]        = (presetValue | occaKeywordType);
      cKeywordType["occaLocalMemFence"]  = (presetValue | occaKeywordType);
      cKeywordType["occaGlobalMemFence"] = (presetValue | occaKeywordType);

      cKeywordType["occaInnerFor0"] = occaForType;
      cKeywordType["occaInnerFor1"] = occaForType;
      cKeywordType["occaInnerFor2"] = occaForType;

      cKeywordType["occaOuterFor0"] = occaForType;
      cKeywordType["occaOuterFor1"] = occaForType;
      cKeywordType["occaOuterFor2"] = occaForType;

      cKeywordType["occaInnerId0"] = (presetValue | occaKeywordType);
      cKeywordType["occaInnerId1"] = (presetValue | occaKeywordType);
      cKeywordType["occaInnerId2"] = (presetValue | occaKeywordType);

      cKeywordType["occaOuterId0"] = (presetValue | occaKeywordType);
      cKeywordType["occaOuterId1"] = (presetValue | occaKeywordType);
      cKeywordType["occaOuterId2"] = (presetValue | occaKeywordType);

      cKeywordType["occaGlobalId0"] = (presetValue | occaKeywordType);
      cKeywordType["occaGlobalId1"] = (presetValue | occaKeywordType);
      cKeywordType["occaGlobalId2"] = (presetValue | occaKeywordType);

      cKeywordType["occaInnerDim0"] = (presetValue | occaKeywordType);
      cKeywordType["occaInnerDim1"] = (presetValue | occaKeywordType);
      cKeywordType["occaInnerDim2"] = (presetValue | occaKeywordType);

      cKeywordType["occaOuterDim0"] = (presetValue | occaKeywordType);
      cKeywordType["occaOuterDim1"] = (presetValue | occaKeywordType);
      cKeywordType["occaOuterDim2"] = (presetValue | occaKeywordType);

      cKeywordType["occaGlobalDim0"] = (presetValue | occaKeywordType);
      cKeywordType["occaGlobalDim1"] = (presetValue | occaKeywordType);
      cKeywordType["occaGlobalDim2"] = (presetValue | occaKeywordType);

      //---[ CUDA Keywords ]--------------
      cKeywordType["threadIdx"] = (unknownVariable | cudaKeywordType);
      cKeywordType["blockDim"]  = (unknownVariable | cudaKeywordType);
      cKeywordType["blockIdx"]  = (unknownVariable | cudaKeywordType);
      cKeywordType["gridDim"]   = (unknownVariable | cudaKeywordType);

      std::string mathFunctions[16] = {
        "sqrt", "sin"  , "asin" ,
        "sinh", "asinh", "cos"  ,
        "acos", "cosh" , "acosh",
        "tan" , "atan" , "tanh" ,
        "atanh", "exp" , "log2" ,
        "log10"
      };

      for(int i = 0; i < 16; ++i){
        std::string mf = mathFunctions[i];
        std::string cmf = mf;
        cmf[0] += ('A' - 'a');

        cKeywordType["occa"       + cmf] = presetValue;
        cKeywordType["occaFast"   + cmf] = presetValue;
        cKeywordType["occaNative" + cmf] = presetValue;
      }

      //---[ Operator Precedence ]--------
      opPrecedence[opHolder("::", binaryOperatorType)]   = 0;

      // class(...), class{1,2,3}, static_cast<>(), func(), arr[]
      opPrecedence[opHolder("++", rUnitaryOperatorType)] = 1;
      opPrecedence[opHolder("--", rUnitaryOperatorType)] = 1;
      opPrecedence[opHolder("." , binaryOperatorType)]   = 1;
      opPrecedence[opHolder("->", binaryOperatorType)]   = 1;

      // (int) x, sizeof, new, new [], delete, delete []
      opPrecedence[opHolder("++", lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("--", lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("+" , lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("-" , lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("!" , lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("~" , lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("*" , qualifierType)]        = 2;
      opPrecedence[opHolder("&" , qualifierType)]        = 2;

      opPrecedence[opHolder(".*" , binaryOperatorType)]  = 3;
      opPrecedence[opHolder("->*", binaryOperatorType)]  = 3;

      opPrecedence[opHolder("*" , binaryOperatorType)]   = 4;
      opPrecedence[opHolder("/" , binaryOperatorType)]   = 4;
      opPrecedence[opHolder("%" , binaryOperatorType)]   = 4;

      opPrecedence[opHolder("+" , binaryOperatorType)]   = 5;
      opPrecedence[opHolder("-" , binaryOperatorType)]   = 5;

      opPrecedence[opHolder("<<", binaryOperatorType)]   = 6;
      opPrecedence[opHolder(">>", binaryOperatorType)]   = 6;

      opPrecedence[opHolder("<" , binaryOperatorType)]   = 7;
      opPrecedence[opHolder("<=", binaryOperatorType)]   = 7;
      opPrecedence[opHolder(">=", binaryOperatorType)]   = 7;
      opPrecedence[opHolder(">" , binaryOperatorType)]   = 7;

      opPrecedence[opHolder("==", binaryOperatorType)]   = 8;
      opPrecedence[opHolder("!=", binaryOperatorType)]   = 8;

      opPrecedence[opHolder("&" , binaryOperatorType)]   = 9;

      opPrecedence[opHolder("^" , binaryOperatorType)]   = 10;

      opPrecedence[opHolder("|" , binaryOperatorType)]   = 11;

      opPrecedence[opHolder("&&", binaryOperatorType)]   = 12;

      opPrecedence[opHolder("||", binaryOperatorType)]   = 13;

      opPrecedence[opHolder("?" , ternaryOperatorType)]  = 14;
      opPrecedence[opHolder("=" , assOperatorType)]      = 14;
      opPrecedence[opHolder("+=", assOperatorType)]      = 14;
      opPrecedence[opHolder("-=", assOperatorType)]      = 14;
      opPrecedence[opHolder("*=", assOperatorType)]      = 14;
      opPrecedence[opHolder("/=", assOperatorType)]      = 14;
      opPrecedence[opHolder("%=", assOperatorType)]      = 14;
      opPrecedence[opHolder("<<=", assOperatorType)]     = 14;
      opPrecedence[opHolder(">>=", assOperatorType)]     = 14;
      opPrecedence[opHolder("&=", assOperatorType)]      = 14;
      opPrecedence[opHolder("^=", assOperatorType)]      = 14;
      opPrecedence[opHolder("|=", assOperatorType)]      = 14;

      // 15: throw x

      opPrecedence[opHolder("," , binaryOperatorType)]   = 16;

      opLevelMap[ 0]["::"]  = binaryOperatorType;
      opLevelMap[ 1]["++"]  = rUnitaryOperatorType;
      opLevelMap[ 1]["--"]  = rUnitaryOperatorType;
      opLevelMap[ 1]["." ]  = binaryOperatorType;
      opLevelMap[ 1]["->"]  = binaryOperatorType;
      opLevelMap[ 2]["++"]  = lUnitaryOperatorType;
      opLevelMap[ 2]["--"]  = lUnitaryOperatorType;
      opLevelMap[ 2]["+" ]  = lUnitaryOperatorType;
      opLevelMap[ 2]["-" ]  = lUnitaryOperatorType;
      opLevelMap[ 2]["!" ]  = lUnitaryOperatorType;
      opLevelMap[ 2]["~" ]  = lUnitaryOperatorType;
      opLevelMap[ 2]["*" ]  = qualifierType;
      opLevelMap[ 2]["&" ]  = qualifierType;
      opLevelMap[ 3][".*" ] = binaryOperatorType;
      opLevelMap[ 3]["->*"] = binaryOperatorType;
      opLevelMap[ 4]["*" ]  = binaryOperatorType;
      opLevelMap[ 4]["/" ]  = binaryOperatorType;
      opLevelMap[ 4]["%" ]  = binaryOperatorType;
      opLevelMap[ 5]["+" ]  = binaryOperatorType;
      opLevelMap[ 5]["-" ]  = binaryOperatorType;
      opLevelMap[ 6]["<<"]  = binaryOperatorType;
      opLevelMap[ 6][">>"]  = binaryOperatorType;
      opLevelMap[ 7]["<" ]  = binaryOperatorType;
      opLevelMap[ 7]["<="]  = binaryOperatorType;
      opLevelMap[ 7][">="]  = binaryOperatorType;
      opLevelMap[ 7][">" ]  = binaryOperatorType;
      opLevelMap[ 8]["=="]  = binaryOperatorType;
      opLevelMap[ 8]["!="]  = binaryOperatorType;
      opLevelMap[ 9]["&" ]  = binaryOperatorType;
      opLevelMap[10]["^" ]  = binaryOperatorType;
      opLevelMap[11]["|" ]  = binaryOperatorType;
      opLevelMap[12]["&&"]  = binaryOperatorType;
      opLevelMap[13]["||"]  = binaryOperatorType;
      opLevelMap[14]["?" ]  = ternaryOperatorType;
      opLevelMap[14]["=" ]  = assOperatorType;
      opLevelMap[14]["+="]  = assOperatorType;
      opLevelMap[14]["-="]  = assOperatorType;
      opLevelMap[14]["*="]  = assOperatorType;
      opLevelMap[14]["/="]  = assOperatorType;
      opLevelMap[14]["%="]  = assOperatorType;
      opLevelMap[14]["<<="] = assOperatorType;
      opLevelMap[14][">>="] = assOperatorType;
      opLevelMap[14]["&="]  = assOperatorType;
      opLevelMap[14]["^="]  = assOperatorType;
      opLevelMap[14]["|="]  = assOperatorType;
      opLevelMap[16][","]   = binaryOperatorType;

      /*---[ Future Ones ]----------------
        cKeywordType["using"] = ;
        cKeywordType["namespace"] = ;
        cKeywordType["template"] = ;
        ================================*/

      keywordType = cKeywordType;
    }

    void initFortranKeywords(){
      if(fortranKeywordsAreInitialized){
        keywordType = fortranKeywordType;
        return;
      }

      fortranKeywordsAreInitialized = true;

      //---[ Operator Info ]--------------
      fortranKeywordType["%"]  = binaryOperatorType;
      fortranKeywordType["("]  = startParentheses;
      fortranKeywordType[")"]  = endParentheses;
      fortranKeywordType["(/"] = startParentheses;
      fortranKeywordType["/)"] = endParentheses;
      fortranKeywordType["**"] = (binaryOperatorType);
      fortranKeywordType["*"]  = (binaryOperatorType);
      fortranKeywordType["+"]  = (lUnitaryOperatorType | binaryOperatorType);
      fortranKeywordType[","]  = binaryOperatorType;
      fortranKeywordType["-"]  = (lUnitaryOperatorType | binaryOperatorType);
      fortranKeywordType["/"]  = binaryOperatorType;
      fortranKeywordType[";"]  = endStatement;
      fortranKeywordType["<"]  = binaryOperatorType;
      fortranKeywordType["="]  = binaryOperatorType;
      fortranKeywordType[">"]  = binaryOperatorType;
      fortranKeywordType["=>"] = binaryOperatorType;
      fortranKeywordType["::"] = binaryOperatorType;
      fortranKeywordType["<="] = binaryOperatorType;
      fortranKeywordType["=="] = binaryOperatorType;
      fortranKeywordType["/="] = binaryOperatorType;
      fortranKeywordType[">="] = binaryOperatorType;

      fortranKeywordType["#"]  = macroKeywordType;

      //---[ Types & Specifiers ]---------
      fortranKeywordType["int"]    = specifierType;
      fortranKeywordType["bool"]   = specifierType;
      fortranKeywordType["char"]   = specifierType;
      fortranKeywordType["long"]   = specifierType;
      fortranKeywordType["short"]  = specifierType;
      fortranKeywordType["float"]  = specifierType;
      fortranKeywordType["double"] = specifierType;

      fortranKeywordType["void"]   = specifierType;

      fortranKeywordType["true"]  = presetValue;
      fortranKeywordType["false"] = presetValue;

      fortranKeywordType["INTEGER"]    = specifierType;
      fortranKeywordType["LOGICAL"]    = specifierType;
      fortranKeywordType["REAL"]       = specifierType;
      fortranKeywordType["PRECISION"]  = specifierType;
      fortranKeywordType["COMPLEX"]    = specifierType;
      fortranKeywordType["CHARACTER"]  = specifierType;

      fortranKeywordType["FUNCTION"]   = specialKeywordType;
      fortranKeywordType["SUBROUTINE"] = specialKeywordType;
      fortranKeywordType["CALL"]       = specialKeywordType;

      fortranKeywordType["DOUBLE"] = qualifierType;

      fortranKeywordType["ALLOCATABLE"] = qualifierType;
      fortranKeywordType["AUTOMATIC"]   = qualifierType;
      fortranKeywordType["DIMENSION"]   = qualifierType;
      fortranKeywordType["EXTERNAL"]    = qualifierType;
      fortranKeywordType["IMPLICIT"]    = qualifierType;
      fortranKeywordType["INTENT"]      = qualifierType;
      fortranKeywordType["INTRINSIC"]   = qualifierType;
      fortranKeywordType["OPTIONAL"]    = qualifierType;
      fortranKeywordType["PARAMETER"]   = qualifierType;
      fortranKeywordType["POINTER"]     = qualifierType;
      fortranKeywordType["PRIVATE"]     = qualifierType;
      fortranKeywordType["PUBLIC"]      = qualifierType;
      fortranKeywordType["RECURSIVE"]   = qualifierType;
      fortranKeywordType["SAVE"]        = qualifierType;
      fortranKeywordType["STATIC"]      = qualifierType;
      fortranKeywordType["TARGET"]      = qualifierType;
      fortranKeywordType["VOLATILE"]    = qualifierType;

      fortranKeywordType["NONE"] = specialKeywordType;

      fortranKeywordType["KERNEL"]    = qualifierType;
      fortranKeywordType["DEVICE"]    = qualifierType;
      fortranKeywordType["SHARED"]    = qualifierType;
      fortranKeywordType["EXCLUSIVE"] = qualifierType;

      //---[ Constants ]------------------
      fortranKeywordType[":"]       = presetValue;

      //---[ Flow Control ]---------------
      fortranKeywordType["DO"]       = flowControlType;
      fortranKeywordType["WHILE"]    = flowControlType;
      fortranKeywordType["DO WHILE"] = flowControlType;

      fortranKeywordType["IF"]       = flowControlType;
      fortranKeywordType["THEN"]     = flowControlType;
      fortranKeywordType["ELSE IF"]  = flowControlType;
      fortranKeywordType["ELSE"]     = flowControlType;

      fortranKeywordType["ENDDO"]         = endStatement;
      fortranKeywordType["ENDIF"]         = endStatement;
      fortranKeywordType["ENDFUNCTION"]   = endStatement;
      fortranKeywordType["ENDSUBROUTINE"] = endStatement;

      std::string mathFunctions[16] = {
        "SQRT", "SIN"  , "ASIN" ,
        "SINH", "ASINH", "COS"  ,
        "ACOS", "COSH" , "ACOSH",
        "TAN" , "ATAN" , "TANH" ,
        "ATANH", "EXP" , "LOG2" ,
        "LOG10"
      };

      //---[ Operator Precedence ]--------
      opPrecedence[opHolder("**", binaryOperatorType)]     = 0;
      opPrecedence[opHolder("//", binaryOperatorType)]     = 0;

      opPrecedence[opHolder("+", lUnitaryOperatorType)]    = 1;
      opPrecedence[opHolder("-", lUnitaryOperatorType)]    = 1;

      opPrecedence[opHolder("*", binaryOperatorType)]      = 2;
      opPrecedence[opHolder("/", binaryOperatorType)]      = 2;

      opPrecedence[opHolder("+", binaryOperatorType)]      = 3;
      opPrecedence[opHolder("-", binaryOperatorType)]      = 3;

      opPrecedence[opHolder("<" , binaryOperatorType)]     = 4;
      opPrecedence[opHolder("<=", binaryOperatorType)]     = 4;
      opPrecedence[opHolder(">=", binaryOperatorType)]     = 4;
      opPrecedence[opHolder(">" , binaryOperatorType)]     = 4;

      opPrecedence[opHolder("!", binaryOperatorType)]      = 5;
      opPrecedence[opHolder("&&", binaryOperatorType)]     = 6;
      opPrecedence[opHolder("||", binaryOperatorType)]     = 7;

      opPrecedence[opHolder("==" , binaryOperatorType)]    = 8;
      opPrecedence[opHolder("!=", binaryOperatorType)]     = 8;

      opPrecedence[opHolder("=" , binaryOperatorType)]     = 9;

      opPrecedence[opHolder("," , binaryOperatorType)]     = 10;

      opLevelMap[0]["**"]  = binaryOperatorType;
      opLevelMap[0]["//"]  = binaryOperatorType;
      opLevelMap[1]["+"]   = lUnitaryOperatorType;
      opLevelMap[1]["-"]   = lUnitaryOperatorType;
      opLevelMap[2]["*"]   = binaryOperatorType;
      opLevelMap[2]["/"]   = binaryOperatorType;
      opLevelMap[3]["+"]   = binaryOperatorType;
      opLevelMap[3]["-"]   = binaryOperatorType;
      opLevelMap[4]["<"]   = binaryOperatorType;
      opLevelMap[4]["<="]  = binaryOperatorType;
      opLevelMap[4][">="]  = binaryOperatorType;
      opLevelMap[4][">"]   = binaryOperatorType;
      opLevelMap[5]["!"]   = binaryOperatorType;
      opLevelMap[6]["&&"]  = binaryOperatorType;
      opLevelMap[7]["||"]  = binaryOperatorType;
      opLevelMap[8]["=="]  = binaryOperatorType;
      opLevelMap[8]["!="]  = binaryOperatorType;
      opLevelMap[9]["="]   = binaryOperatorType;
      opLevelMap[10][","]  = binaryOperatorType;

      keywordType = fortranKeywordType;
    }

    //---[ OCCA Loop Info ]-------------
    occaLoopInfo::occaLoopInfo(statement &s,
                               const bool parsingC_,
                               const std::string &tag){
      parsingC = parsingC_;

      lookForLoopFrom(s, tag);
    }

    void occaLoopInfo::lookForLoopFrom(statement &s,
                                       const std::string &tag){
      sInfo = &s;

      while(sInfo){
        if((sInfo->info & forStatementType) &&
           (sInfo->getForStatementCount() > 3)){

          if(4 < sInfo->getForStatementCount()){
            std::cout << "More than 4 statements for:\n  " << sInfo->expRoot << '\n';
            throw 1;
          }

          if(tag.size()){
            std::string arg4 = (std::string) *(sInfo->getForStatement(3));

            if(arg4 == tag)
              break;
          }
          else
            break;
        }

        sInfo = sInfo->up;
      }

      //---[ Overload iter vars ]---
      setIterDefaultValues();

      sInfo->info = occaForType;

      expNode &node1   = *(sInfo->getForStatement(0));
      expNode &node2   = *(sInfo->getForStatement(1));
      expNode &node3   = *(sInfo->getForStatement(2));
      std::string arg4 = (std::string) *(sInfo->getForStatement(3));

      // Fortran-loading is easier
      if(!parsingC){
        //---[ Node 4 Check ]---
        // If it has a fourth argument, make sure it's the correct one
        if( !isAnOccaTag(arg4) ){
          std::cout << "Wrong 4th statement for:\n  " << sInfo->expRoot << '\n';
          throw 1;
        }

        return;
      }

      //---[ Node 1 Check ]---
      if((node1.info != expType::declaration) ||
         (node1.getVariableCount() != 1)      ||
         !node1.variableHasInit(0)){

        std::cout << "Wrong 1st statement for:\n  " << sInfo->expRoot << '\n';
        throw 1;
      }

      varInfo &iterVar = node1.getVariableInfoNode(0)->getVarInfo();

      std::string &iter = iterVar.name;

      if( !iterVar.hasQualifier("occaConst") )
        iterVar.addQualifier("occaConst");

      //---[ Node 2 Check ]---
      if((node2.leafCount != 1) ||
         ((node2[0].value != "<=") &&
          (node2[0].value != "<" ) &&
          (node2[0].value != ">" ) &&
          (node2[0].value != ">="))){

        std::cout << "Wrong 2nd statement for:\n  " << sInfo->expRoot << '\n';
        throw 1;
      }

      if(parsingC){
        if((node2[0][0].value != iter) &&
           (node2[0][1].value != iter)){

          std::cout << "Wrong 2nd statement for:\n  " << sInfo->expRoot << '\n';
          throw 1;
        }
      }

      //---[ Node 3 Check ]---
      if((node3.leafCount != 1) ||
         ((node3[0].value != "++") &&
          (node3[0].value != "--") &&
          (node3[0].value != "+=") &&
          (node3[0].value != "-="))){

        std::cout << "Wrong 3nd statement for:\n  " << sInfo->expRoot << '\n';
        throw 1;
      }

      if((node3[0][0].value != iter) &&
         (node3[0][1].value != iter)){

        std::cout << "Wrong 3rd statement for:\n  " << sInfo->expRoot << '\n';
        throw 1;
      }

      //---[ Node 4 Check ]---
      // If it has a fourth argument, make sure it's the correct one
      if( !isAnOccaTag(arg4) ){
        std::cout << "Wrong 4th statement for:\n  " << sInfo->expRoot << '\n';
        throw 1;
      }
    }

    // [-] Missing
    void occaLoopInfo::loadForLoopInfo(int &innerDims, int &outerDims,
                                       std::string *innerIters,
                                       std::string *outerIters){
    }

    void occaLoopInfo::getLoopInfo(std::string &ioLoopVar,
                                   std::string &ioLoop,
                                   std::string &loopNest){
      std::string arg4 = (std::string) *(sInfo->getForStatement(3));

      // [-----][#]
      ioLoopVar = arg4.substr(0,5);
      ioLoop    = ioLoopVar;
      loopNest  = arg4.substr(5,1);

      ioLoop[0] += ('A' - 'a');
    }

    void occaLoopInfo::getLoopNode1Info(std::string &iter,
                                        std::string &start){
      expNode &node1 = *(sInfo->getForStatement(0));

      if(parsingC){
        varInfo &iterVar = node1.getVariableInfoNode(0)->getVarInfo();

        iter  = iterVar.name;
        start = *(node1.getVariableInitNode(0));
      }
      else{
        iter  = node1[0][0].value;
        start = node1[0][1].value;
      }
    }

    void occaLoopInfo::getLoopNode2Info(std::string &bound,
                                        std::string &iterCheck){
      expNode &node2 = *(sInfo->getForStatement(1));

      iterCheck = node2[0].value;

      if(parsingC){
        if((iterCheck == "<=") || (iterCheck == "<"))
          bound = (std::string) node2[0][1];
        else
          bound = (std::string) node2[0][0];
      }
      else{
        std::string iter, start;
        getLoopNode1Info(iter, start);

        // [doStart][#]
        std::string suffix = start.substr(7);

        bound = "(doEnd" + suffix + " + 1)";
      }
    }

    void occaLoopInfo::getLoopNode3Info(std::string &stride,
                                        std::string &strideOpSign,
                                        std::string &strideOp){
      expNode &node3 = *(sInfo->getForStatement(2));

      std::string iter, tmp;
      getLoopNode1Info(iter, tmp);

      strideOp = node3[0].value;

      // [+]+, [+]=
      // [-]-, [-]=
      strideOpSign = strideOp[0];

      if((strideOp == "++") || (strideOp == "--")){
        stride = "1";
      }
      else{
        if(node3[0][0].value == iter)
          stride = (std::string) node3[0][1];
        else
          stride = (std::string) node3[0][0];
      }
    }

    void occaLoopInfo::setIterDefaultValues(){
      int innerDims, outerDims;
      std::string innerIters[3], outerIters[3];

      loadForLoopInfo(innerDims, outerDims,
                      innerIters, outerIters);
    }

    std::string occaLoopInfo::getSetupExpression(){
      expNode &node1 = *(sInfo->getForStatement(0));

      std::stringstream ss;
      node1.printOn(ss, "", expFlag::noSemicolon);

      return ss.str();
    }
    //==================================
  };
};
