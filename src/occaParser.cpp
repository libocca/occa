#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    parserBase::parserBase(){
      macrosAreInitialized = false;
      globalScope = new statement(*this);
    }

    const std::string parserBase::parseFile(const std::string &filename){
      initMacros();

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

      globalScope->loadAllFromNode(nodeRoot);
      // std::cout << (std::string) *globalScope;
      // throw 1;

      markKernelFunctions(*globalScope);
      applyToAllStatements(*globalScope, &parserBase::labelKernelsAsNativeOrNot);

      applyToAllStatements(*globalScope, &parserBase::setupCudaVariables);
      applyToAllStatements(*globalScope, &parserBase::setupOccaVariables);

      // Broken?
      // fixOccaForOrder(); // + auto-adds barriers

      // Broken
      addFunctionPrototypes();
      // Broken
      applyToAllStatements(*globalScope, &parserBase::updateConstToConstant);

      addOccaFors();

      applyToAllStatements(*globalScope, &parserBase::addParallelFors);
      applyToAllStatements(*globalScope, &parserBase::setupOccaFors);

      applyToAllStatements(*globalScope, &parserBase::modifyExclusiveVariables);
      // Broken
      modifyTextureVariables();

      addArgQualifiers();

      loadKernelInfos();

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
        if(labelNodePos->type & unknownVariable)
          return false;

        labelNodePos = labelNodePos->right;
      }

      typeHolder th = evaluateLabelNode(labelNodeRoot);

      return (th.doubleValue() != 0);
    }

    typeHolder parserBase::evaluateLabelNode(strNode *labelNodeRoot){
      if(labelNodeRoot->type & presetValue)
        return typeHolder(*labelNodeRoot);

      strNode *labelNodePos = labelNodeRoot;

      while(labelNodePos){
        if(labelNodePos->down){
          labelNodePos->value = evaluateLabelNode(labelNodePos->down);
          labelNodePos->type  = presetValue;
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
          if(labelNodePos->type & operatorType){
            int opType = (labelNodePos->type & operatorType);

            opType &= ~qualifierType;

            if(opType & unitaryOperatorType){
              if((opType & binaryOperatorType) && // + and - operators
                 (labelNodePos->left)          &&
                 (labelNodePos->left->type & presetValue)){

                opType = binaryOperatorType;
              }
              else if((opType & rUnitaryOperatorType) &&
                      (labelNodePos->left)            &&
                      (labelNodePos->left->type & presetValue)){

                opType = rUnitaryOperatorType;
              }
              else if((opType & lUnitaryOperatorType) &&
                      (labelNodePos->right)           &&
                      (labelNodePos->right->type & presetValue)){

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
              minOpNode->type  = presetValue;

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
            minOpNode->type  = presetValue;

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
            minOpNode->type  = presetValue;

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

          if(currentState & keepMacro){
            currentState &= ~keepMacro;

            if( !(currentState & ignoring) )
              nodePos->type = macroKeywordType;
          }
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
        else
          nodePos = nodePos->right;
      }

      return nodeRoot;
    }

    strNode* parserBase::splitAndPreprocessContent(const std::string &s){
      return splitAndPreprocessContent(s.c_str());
    }

    strNode* parserBase::splitAndPreprocessContent(const char *cRoot){
      initKeywords();
      initMacros();

      strNode *nodeRoot = splitContent(cRoot);

      nodeRoot = preprocessMacros(nodeRoot);
      nodeRoot = labelCode(nodeRoot);

      return nodeRoot;
    }
    //====================================

    void parserBase::initMacros(){
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
      if(s.type & functionStatementType){
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
      statement *ret = ((s.type == keywordType["occaOuterFor0"]) ? &s : NULL);

      statement *sUp = &s;

      while(sUp){
        if(sUp->type == keywordType["occaOuterFor0"])
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

    bool parserBase::nodeHasUnknownVariable(strNode *n){
      return false;
#if 0
      while(n){
        if(n->type & unknownVariable)
          return true;

        const int downCount = n->down.size();

        for(int i = 0; i < downCount; ++i)
          if(nodeHasUnknownVariable(n->down[i]))
            return true;

        n = n->right;
      }

      return false;
#endif
    }

    void parserBase::setupOccaFors(statement &s){
      if( !(s.type & forStatementType) )
        return;

      statement *spKernel = getStatementKernel(s);

      if(spKernel == NULL)
        return;

      if(statementKernelUsesNativeOCCA(s))
        return;

      if(s.getForStatementCount() <= 3)
        return;

      if(4 < s.getForStatementCount()){
        std::cout << "More than 4 statements for:\n  " << s.expRoot << '\n';
        throw 1;
      }

      s.type = keywordType["occaOuterFor0"];

      statement &sKernel = *spKernel;

      std::string arg4 = (std::string) *(s.expRoot.leaves[3]);

      // If it has a fourth argument, make sure it's the correct one
      if( ((arg4.find("inner") == std::string::npos) ||
           ((arg4 != "inner0") &&
            (arg4 != "inner1") &&
            (arg4 != "inner2")))                     &&
          ((arg4.find("outer") == std::string::npos) ||
           ((arg4 != "outer0") &&
            (arg4 != "outer1") &&
            (arg4 != "outer2"))) ){

        std::cout << "Wrong 4th statement for:\n  " << s.expRoot << '\n';
        throw 1;
      }

      // [-----][#]
      std::string ioLoopVar = arg4.substr(0,5);
      std::string ioLoop    = ioLoopVar;
      std::string loopNest  = arg4.substr(5,1);

      ioLoop[0] += ('A' - 'a');

      //---[ Find operators ]-----------
      std::string iter, start, bound, stride;
      std::string iterCheck, iterOp;
      std::string opSign, opStride;

      expNode &node1 = *(s.expRoot.leaves[0]);
      expNode &node2 = *(s.expRoot.leaves[1]);
      expNode &node3 = *(s.expRoot.leaves[2]);

      //---[ Node 1 ]---------
      if((node1.info != expType::declaration) ||
         (node1.getVariableCount() != 1)      ||
         !node1.variableHasInit(0)){
        std::cout << "Wrong 1st statement for:\n  " << s.expRoot << '\n';
        throw 1;
      }

      varInfo &iterVar = node1.getVariableInfoNode(0)->getVarInfo();

      if( !iterVar.hasQualifier("occaConst") )
        iterVar.addQualifier("occaConst");

      iter  = iterVar.name;
      start = *(node1.getVariableInitNode(0));

      //---[ Node 2 ]---------
      if((node2.leafCount != 1) ||
         ((node2[0].value != "<=") &&
          (node2[0].value != "<" ) &&
          (node2[0].value != ">" ) &&
          (node2[0].value != ">="))){

        std::cout << "Wrong 2nd statement for:\n  " << s.expRoot << '\n';
        throw 1;
      }

      if(node2[0][0].value == iter){
        bound = (std::string) node2[0][1];
      }
      else if(node2[0][1].value == iter){
        bound = (std::string) node2[0][0];
      }
      else {
        std::cout << "Wrong 2nd statement for:\n  " << s.expRoot << '\n';
        throw 1;
      }

      iterCheck = node2[0].value;

      //---[ Node 3 ]---------
      if((node3.leafCount != 1) ||
         ((node3[0].value != "++") &&
          (node3[0].value != "--") &&
          (node3[0].value != "+=") &&
          (node3[0].value != "-="))){
        std::cout << "Wrong 3nd statement for:\n  " << s.expRoot << '\n';
        throw 1;
      }

      iterOp = node3[0].value;

      // [+]+, [+]=
      // [-]-, [-]=
      opSign = iterOp[0];

      if((iterOp == "++") || (iterOp == "--"))
        opStride = "1";
      else{
        if(node3[0][0].value == iter){
          opStride = (std::string) node3[0][1];
        }
        else if(node3[0][1].value == iter){
          opStride = (std::string) node3[0][0];
        }
        else {
          std::cout << "Wrong 3rd statement for:\n  " << s.expRoot << '\n';
          throw 1;
        }
      }

      if(opSign[0] == '-')
        stride  = "-(";
      else
        stride  = "(";

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
        node1.printOn(ss, "", expFlag::noSemicolon);

        ss << ' '
           << opSign
           << " (occa" << ioLoop << "Id" << loopNest
           << " * (" << opStride << "));";
      }
      else{
        node1.printOn(ss, "", expFlag::noSemicolon);

        ss << ' '
           << opSign
           << " occa" << ioLoop << "Id" << loopNest << ";";
      }

      s.scopeVarMap.erase(iter);

      s.pushLeftFromSource(s.statementStart, ss.str());

      std::string occaForName = "occa" + ioLoop + "For" + loopNest;

      s.expRoot.info  = expType::occaFor;
      s.expRoot.value = occaForName;
      s.expRoot.free();
    }

    bool parserBase::statementHasOccaOuterFor(statement &s){
      if(s.type == keywordType["occaOuterFor0"]){
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
      if(s.type == keywordType["occaOuterFor0"])
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
      if(s.type == forStatementType)
        return (s.expRoot.leafCount == 4);

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

        if( !(s2.type & functionStatementType) ||
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

    void parserBase::labelKernelsAsNativeOrNot(statement &s){
      if(!statementIsAKernel(s))
        return;

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

    void parserBase::setupCudaVariables(statement &s){
      if((!(s.type & simpleStatementType)    &&
          !(s.type & forStatementType)       &&
          !(s.type & functionStatementType)) ||
         // OCCA for's don't have arguments
         (s.type == keywordType["occaOuterFor0"]))
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
#if 0
      std::map<std::string,bool> prototypes;

      statementNode *statementPos = globalScope->statementStart;

      while(statementPos){
        statement *s2 = statementPos->value;

        if(s2->type & functionPrototypeType)
          prototypes[s2->getFunctionName()] = true;

        statementPos = statementPos->right;
      }

      statementPos = globalScope->statementStart;

      while(statementPos){
        statement *s2 = statementPos->value;

        if(s2->type & functionStatementType){
          if(s2->hasQualifier("occaKernel")){
            statementPos = statementPos->right;
            continue;
          }

          if(!s2->hasQualifier("occaFunction"))
            s2->addQualifier("occaFunction");

          if( !(s2->type & functionDefinitionType) ){
            statementPos = statementPos->right;
            continue;
          }

          if(prototypes.find( s2->getFunctionName() ) == prototypes.end()){
            statement *newS2 = s2->clone();
            statementNode *newNode = new statementNode(newS2);

            newS2->type = functionPrototypeType;

            newS2->statementCount = 0;

            // [-] Delete definition (needs proper free)
            delete newS2->statementStart;
            newS2->statementStart = NULL;
            newS2->statementEnd   = NULL;

            strNode *end = newS2->nodeEnd;
            end = end->push(";");

            end->up        = newS2->nodeEnd->up;
            end->type      = keywordType[";"];
            end->depth     = newS2->nodeEnd->depth;
            end->sideDepth = newS2->nodeEnd->sideDepth;

            newS2->nodeEnd = end;

            statementNode *left = statementPos->left;

            if(globalScope->statementStart == statementPos)
              globalScope->statementStart = newNode;

            if(left)
              left->right = newNode;

            newNode->left = left;

            newNode->right     = statementPos;
            statementPos->left = newNode;

            ++(globalScope->statementCount);
          }
        }

        statementPos = statementPos->right;
      }
#endif
    }

    int parserBase::statementOccaForNest(statement &s){
      if((s.type != forStatementType) ||
         (s.getForStatementCount() != 4)){

        return notAnOccaFor;
      }

      int ret = notAnOccaFor;

      expNode &labelNode = *(s.getForStatement(3));

      const std::string &forName = (std::string) labelNode;

      if((forName.find("outer") != std::string::npos) &&
         ((forName == "outer0") ||
          (forName == "outer1") ||
          (forName == "outer2"))){

        ret = ((1 + forName[5] - '0') << occaOuterForShift);
      }
      else if((forName.find("inner") != std::string::npos) &&
              ((forName == "inner0") ||
               (forName == "inner1") ||
               (forName == "inner2"))){

        ret = ((1 + forName[5] - '0') << occaInnerForShift);
      }

      return ret;
    }

    bool parserBase::statementIsAnOccaFor(statement &s){
      const int nest = statementOccaForNest(s);

      return !(nest & notAnOccaFor);
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

            if(!statementHasBarrier( *(sn->left->value) )){
              std::cout << "Warning: Placing a local barrier between:\n"
                        << "---[ A ]--------------------------------\n"
                        << *(sn->left->value)
                        << "---[ B ]--------------------------------\n"
                        << *(sn->value)
                        << "========================================\n";

              s.loadFromNode(labelCode( splitContent("occaBarrier(occaLocalMemFence);\0") ));

              statement *newS2     = s.statementEnd->value;
              statementNode *newSN = new statementNode(newS2);

              s.statementEnd        = s.statementEnd->left;
              s.statementEnd->right = NULL;

              newSN->right     = s.statementStart;
              s.statementStart = newSN;

              if(sn->left)
                sn->left->right = newSN;

              newSN->left  = sn->left;
              newSN->right = sn;

              sn->left = newSN;
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

    void parserBase::addParallelFors(statement &s){
      if( !statementIsAKernel(s) )
        return;

      statementNode *snPos = s.statementStart;

      while(snPos){
        statement &s2 = *(snPos->value);

        const int nest = statementOccaForNest(s2);

        if(nest & (notAnOccaFor | occaInnerForMask)){

          snPos = snPos->right;
          continue;
        }

        const char outerDim = '0' + (nest - 1);

        statement *parallelStatement = new statement(s.depth + 1,
                                                     occaStatementType, &s);

        statementNode *parallelSN = new statementNode(parallelStatement);

        parallelStatement->type = macroStatementType;

        parallelStatement->expRoot.value = "occaParallelFor";
        parallelStatement->expRoot.value += outerDim;
        parallelStatement->expRoot.value += '\n';
        parallelStatement->expRoot.info   = expType::occaFor;

        if(s.statementStart == snPos)
          s.statementStart = parallelSN;

        statementNode *leftSN = snPos->left;

        parallelSN->right = snPos;
        parallelSN->left  = leftSN;

        snPos->left = parallelSN->right;

        if(leftSN)
          leftSN->right = parallelSN;

        snPos = snPos->right;
      }
    }

    void parserBase::updateConstToConstant(statement &s){
      return;

      // Global scope only
      if((s.depth != 0) ||
         !(s.type & declareStatementType))
        return;

      strNode *nodePos = s.nodeStart;

      while(nodePos){
        if(nodePos->value == "occaConst")
          nodePos->value = "occaConstant";

        // [*] or [&]
        if(nodePos->type & (unknownVariable |
                            binaryOperatorType))
          break;

        nodePos = nodePos->right;
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

      nodeRoot->type      = presetValue;
      nodeRoot->depth     = depth;
      nodeRoot->sideDepth = sideDepth;

      strNode *nodePos = nodeRoot->pushDown("(");

      nodePos->type  = keywordType["("];
      nodePos->depth = depth + 1;

      var.removeQualifier("exclusive");

      for(int i = 0; i < var.leftQualifiers.qualifierCount; ++i){
        nodePos       = nodePos->push(var.leftQualifiers.qualifiers[i]);
        nodePos->type = qualifierType;
      }

      nodePos       = nodePos->push(var.baseType->name);
      nodePos->type = specifierType;

      for(int i = 0; i < var.rightQualifiers.qualifierCount; ++i){
        nodePos       = nodePos->push(var.rightQualifiers.qualifiers[i]);
        nodePos->type = keywordType[var.rightQualifiers.qualifiers[i]];
      }

      nodePos       = nodePos->push(",");
      nodePos->type = keywordType[","];

      nodePos       = nodePos->push(var.name);
      nodePos->type = unknownVariable;

      if(var.stackPointerCount){
        if(1 < var.stackPointerCount){
          std::cout << "Only 1D exclusive variables are currently supported [" << var << "]\n";
          throw 1;
        }

        nodePos       = nodePos->push(",");
        nodePos->type = keywordType[","];

        nodePos       = nodePos->push((std::string) var.stackExpRoots[0]);
        nodePos->type = presetValue;
      }

      nodePos       = nodePos->push(")");
      nodePos->type = keywordType[")"];

      nodePos       = nodePos->push(";");
      nodePos->type = keywordType[";"];

      return nodeRoot;
    }

    void parserBase::addKernelInfo(varInfo &info, statement &s){
#if 0
      if( !(s.type & functionStatementType) ){
        node<strNode*> nNodeRoot = s.nodeStart->getStrNodesWith(info.name);

        node<strNode*> *nNodePos = nNodeRoot.right;

        while(nNodePos){
          strNode *nodePos = nNodePos->value;

          // If we're calling function, not using a function pointer
          if(nodePos->down.size()){
            nodePos = nodePos->down[0];

            if((nodePos->type == startParentheses) &&
               (nodePos->value != "occaKernelInfo")){
              strNode *kia = nodePos->push("occaKernelInfo");

              kia->up        = nodePos->up;
              kia->type      = keywordType["occaKernelInfo"];
              kia->depth     = nodePos->depth;
              kia->sideDepth = nodePos->sideDepth;

              if(kia->right->type != endParentheses){
                strNode *comma = kia->push(",");

                comma->up        = kia->up;
                comma->type      = keywordType[","];
                comma->depth     = kia->depth;
                comma->sideDepth = kia->sideDepth;
              }
            }
          }

          nNodePos = nNodePos->right;
        }
      }
#endif
    }

    void parserBase::addArgQualifiers(){
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos){
        statement &s = *(statementPos->value);

        if((s.type & functionDefinitionType) &&
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
            }
          }

          if(s.getFunctionArgName(0) != "occaKernelInfoArg"){
            varInfo arg0;
            arg0.name = "occaKernelInfoArg";

            s.addFunctionArg(0, arg0);
          }
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::modifyExclusiveVariables(statement &s){
      if( !(s.type & declareStatementType)   ||
          (getStatementKernel(s) == NULL)    ||
          (statementKernelUsesNativeOCCA(s)) ||
          (!s.hasQualifier("exclusive")) ){

        return;
      }

      std::stringstream ss;

      const int argc = s.getDeclarationVarCount();

      //---[ Setup update statement ]---
      expNode &newRoot = *(s.expRoot.clone(s));
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
      return;

      varUsedMapIterator it = varUsedMap.begin();

      while(it != varUsedMap.end()){
        varInfo *infoPtr = it->first;

        if(infoPtr->hasQualifier("texture")){
          statement &os = *(varOriginMap[infoPtr]);

          strNode *osNodePos = os.nodeStart;

          while(osNodePos){
            if(osNodePos->value == infoPtr->name)
              std::cout << "HERE!\n";

            osNodePos = osNodePos->right;
          }

          // First node is just a placeholder
          statementNode *sNodePos = (it->second).right;

          while(sNodePos){
            statement &s = *(sNodePos->value);
            strNode *nodePos = s.nodeStart;

            while(nodePos){
              if((nodePos->type & unknownVariable) &&
                 (nodePos->value == infoPtr->name)){
                // [-] HERE
                std::cout
                  << "2. nodePos = " << *nodePos << '\n';
              }

              nodePos = nodePos->right;
            }

            sNodePos = sNodePos->right;
          }
        }

        ++it;
      }
    }

    statementNode* parserBase::splitKernelStatement(statementNode *sn,
                                                    kernelInfo &info){
      int kernelCount = 0;

      const int occaForType = keywordType["occaOuterFor0"];

      statement &s         = *(sn->value);
      statement &sUp       = *(s.up);
      statementNode *snPos = s.statementStart;

      statement& hostKernel = *(s.clone());
      stripOccaFromKernel(hostKernel);

      // Count sub-kernels
      while(snPos){
        statement &s2 = *(snPos->value);

        if( (s2.type != occaForType) &&
            !(s2.type & macroStatementType) ){

          std::cout << "Only outer-loops are supported at the kernel scope:\n"
                    << s2 << '\n';
          throw 1;
        }

        if(s2.type == occaForType)
          ++kernelCount;

        snPos = snPos->right;
      }

      std::stringstream ss;

      // Find unique baseName
      while(true){
        int k;

        for(k = 0; k < kernelCount; ++k){
          ss << k;

          if(globalScope->hasVariableInScope(info.baseName + ss.str()))
            break;

          ss.str("");
        }

        if(k == kernelCount)
          break;

        info.baseName += '_';
      }

      statementNode *newSNRoot, *newSNEnd;

      info.nestedKernels.clear();

      scopeVarMapIterator it = globalScope->scopeVarMap.find(info.name);

      // Create empty kernels
      for(int k = 0; k < kernelCount; ++k){
        statement &s2 = *(new statement(s.depth,
                                        s.type, globalScope));

        s2.scopeTypeMap = s.scopeTypeMap;
        s2.scopeVarMap  = s.scopeVarMap;

        info.nestedKernels.push_back(&s2);

        s.expRoot.cloneTo(s2.expRoot);

        ss << k;

        varInfo &kVar = *(s2.getFunctionVar());
        kVar.name    += ss.str();

        ss.str("");

        globalScope->addVariable(&kVar, &s2);

        if(k)
          newSNEnd = newSNEnd->push(new statementNode(info.nestedKernels.back()));
        else
          newSNRoot = newSNEnd = new statementNode(info.nestedKernels.back());
      }

      // Squeeze new kernels after original kernel
      sn->right       = newSNRoot;
      newSNRoot->left = sn;

      if(sUp.statementEnd == sn)
        sUp.statementEnd = newSNEnd;

      statementNode *snPosStart = s.statementStart;
      snPos = snPosStart;

      kernelCount = 0;
      int snCount = 0;

      statement &sKernel = *(getStatementKernel(s));

      // occaKernelInfoArg doesn't count
      const int argc = (sKernel.getFunctionArgCount() - 1);
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

      // Add nestedKernels argument
      {
        varInfo arg;
        arg.loadFrom(s, labelCode( splitContent("int *nestedKernels") ));

        typeInfo &type = *(new typeInfo);
        type.name = "occa::kernel";

        arg.baseType = &type;

        s.addFunctionArg(1, arg);
      }

      // Add kernel bodies
      while(snPos){
        statement &s2 = *(snPos->value);

        if((s2.type == occaForType) ||
           (snPos->right == NULL)){

          //---[ Make substitute call ]-----------
          ss << "{\n";

          varInfo *loopIter = s2.hasVariableInScope( obfuscate("loop", "iters") );

          if(loopIter){
            const int extras = loopIter->leftQualifierCount();

            const int outerDim = getOuterMostForDim(s2) + 1;
            const int innerDim = getInnerMostForDim(s2) + 1;

            ss << "  const int dims = " << outerDim << ";\n"
               << "  int outer, inner;\n";

            for(int i = 0; i < (outerDim + innerDim); ++i)
              ss << "  " << loopIter->getLeftQualifier(i) << "\n";

            ss << "  nestedKernels[" << kernelCount << "].setWorkingDims(dims, inner, outer);\n";
          }

          ss << "  nestedKernels[" << kernelCount << "](" << argsStr << ");\n";

          ss << "}";

          s.loadFromNode(labelCode( splitContent(ss.str()) ));

          statementNode *newSN  = s.statementEnd;
          s.statementEnd        = newSN->left;

          if(s.statementEnd)
            s.statementEnd->right = NULL;

          //---[ Change type ]----------
          if(loopIter){
            statement &newS = *(newSN->value);

            varInfo &outerVar = *(newS.hasVariableInScope("outer"));
            statement &outerS = *(varOriginMap[&outerVar]);

            typeInfo &type = *(new typeInfo);
            type.name = "occa::dim";

            outerS.getDeclarationVarInfo(0).baseType = &type;
          }
          //============================

          if(snPosStart){
            newSN->left = snPosStart->left;

            if(newSN->left)
              newSN->left->right = newSN;
          }

          if(snPos){
            newSN->right = snPos->right;

            if(newSN->right)
              newSN->right->left = newSN;
          }

          if(snPosStart == s.statementStart){
            s.statementStart = newSN;
          }
          if(snPos == s.statementEnd){
            s.statementEnd = newSN;
          }

          ss.str("");
          //======================================

          statement &s3 = *(info.nestedKernels[kernelCount]);

          // Stopping at the last node
          if( !(s2.type == keywordType["occaOuterFor0"]) )
            snPos = NULL;

          s3.statementStart = snPosStart;
          s3.statementEnd   = snPos;

          if(snPosStart)
            snPosStart->left = NULL;

          if(snPos){
            statementNode *snPosRight = snPos->right;
            snPos->right = NULL;

            snPos      = snPosRight;
            snPosStart = snPosRight;
          }

          snCount = 0;
          ++kernelCount;
        }
        else{
          snPos = snPos->right;
          ++snCount;
        }

        globalScope->statementCount += (kernelCount - 1);
      }

      // Add kernel guards
      {
        sUp.loadFromNode(labelCode( splitContent("#if OCCA_USING_OPENMP") ));

        statementNode *ifOMP    = sUp.statementEnd;
        sUp.statementEnd        = ifOMP->left;
        sUp.statementEnd->right = NULL;

        ifOMP->left = sn->left;

        if(ifOMP->left)
          ifOMP->left->right = ifOMP;

        ifOMP->right = sn;
        sn->left     = ifOMP;

        if(sUp.statementStart == sn)
          sUp.statementStart = ifOMP;

        sUp.loadFromNode(labelCode( splitContent("#endif") ));

        statementNode *endifOMP = sUp.statementEnd;
        sUp.statementEnd        = endifOMP->left;
        sUp.statementEnd->right = NULL;

        endifOMP->right = sn->right;

        if(endifOMP->right)
          endifOMP->right->left = endifOMP;

        endifOMP->left = sn;
        sn->right      = endifOMP;

        if(sUp.statementEnd == sn)
          sUp.statementEnd = endifOMP;
      }

      return newSNEnd->right;
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
        if(currentS->type == (forStatementType | occaStatementType))
          break;

        currentS = currentS->up;
      }

      if(currentS == NULL)
        return "";

      return s.nodeStart->value;
    }

    void parserBase::incrementDepth(statement &s){
      ++s.depth;
    }

    void parserBase::decrementDepth(statement &s){
      --s.depth;
    }

    bool parserBase::statementHasBarrier(statement &s){
      if(s.expRoot.leafCount == 0)
        return false;

      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        if(flatRoot[i].value == "occaBarrier")
          return true;
      }

      expNode::freeFlatHandle(flatRoot);

      return false;
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

      std::cout << "Error, " << tag << "-most loop doesn't contain obfuscate(\"Outer\"):\n"
                << s.expRoot << '\n';
      return -1;
    }

    void parserBase::checkPathForConditionals(statementNode *path){
      if((path == NULL) ||
         (path->value == NULL))
        return;

      if(path->value->type & ifStatementType){
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
                                     loopSection_t &loopSection,
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
                                      loopSection_t &loopSection){
      // Don't count shared memory
      if(var.hasQualifier("occaPointer")  ||
         var.hasQualifier("occaVariable") ||
         var.hasQualifier("occaShared"))
        return false;

      varUsedMapIterator it = varUsedMap.find(&var);

      // Variable is not used
      if(it == varUsedMap.end())
        return false;

      const int segment = loopSection[varOriginMap[&var]];

      statementNode *pos = (it->second).right;

      while(pos){
        if(segment != loopSection[pos->value])
          return true;

        pos = pos->right;
      }

      return false;
    }

    varInfoNode* parserBase::findVarsMovingToTop(statement &s,
                                                 loopSection_t &loopSection){
      // Statement defines have doubles (to know how many variables
      //                                 were defined)
      //    so ... ignore duplicates
      const bool ignoringVariables = (s.type & simpleStatementType);

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

          s.expRoot.leaves[i]   = origin.expRoot.leaves[i];
          varOriginMap[&argVar] = &s;
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

          s.expRoot.leaves[i]   = origin.expRoot.leaves[argPos + 1 + i];
          varOriginMap[&argVar] = &s;
        }
      }

      origin.expRoot.leaves[0] = origin.expRoot.leaves[argPos];
      origin.expRoot.leafCount = 1;

      if(origin.expRoot.variableHasInit(0)){
        std::stringstream ss;

        ss << var.name << " = " << *(origin.expRoot.getVariableInitNode(0)) << ";";

        origin.up->pushRightFromSource(origin.getStatementNode(),
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
             statementHasBarrier( *(sn->value) )))
        sn = sn->right;

      while(sn){
        sn = addInnerForsBetweenBarriers(s, sn, innerDim);

        while(sn &&
              statementHasBarrier( *(sn->value) ))
          sn = sn->right;
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
        if(statementHasBarrier( *(includeEnd->value) ))
          break;

        // [-] Re-add if we want to split between for-loops
        if(includeEnd->value->type & forStatementType){
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
              forStart->left->value->type & macroStatementType)
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

      loopSection_t loopSection;
      findLoopSections(s, sPath, loopSection);

      // Get private and shared vars
      varInfoNode *varRoot = findVarsMovingToTop(s, loopSection);
      varInfoNode *varPos  = lastNode(varRoot);

      statementNode *newStatementStart = NULL;
      statementNode *newStatementPos   = NULL;

      while(varPos){
        statement &origin = *(varOriginMap[varPos->value]);

        // Ignore kernel arguments
        if(origin.type & functionStatementType){
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

      const int occaForType = keywordType["occaOuterFor0"];

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

        if(sDown->type == blockStatementType){
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

        if((s2.type & declareStatementType) &&
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
      if(s.expRoot.leafCount == 0)
        return;

      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        std::string &value = flatRoot[i].value;
        std::string occaValue;

        const bool isInnerId = ((value == "occaInnerId0") ||
                                (value == "occaInnerId1") ||
                                (value == "occaInnerId2"));

        const bool isOuterId = ((value == "occaOuterId0") ||
                                (value == "occaOuterId1") ||
                                (value == "occaOuterId2"));

        const bool isGlobalId = ((value == "occaGlobalId0") ||
                                 (value == "occaGlobalId1") ||
                                 (value == "occaGlobalId2"));

        const bool isInnerDim = ((value == "occaInnerDim0") ||
                                 (value == "occaInnerDim1") ||
                                 (value == "occaInnerDim2"));

        const bool isOuterDim = ((value == "occaOuterDim0") ||
                                 (value == "occaOuterDim1") ||
                                 (value == "occaOuterDim2"));

        const bool isGlobalDim = ((value == "occaGlobalDim0") ||
                                  (value == "occaGlobalDim1") ||
                                  (value == "occaGlobalDim2"));

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
    //==============================================

    strNode* splitContent(const std::string &str){
      return splitContent(str.c_str());
    }

    strNode* splitContent(const char *cRoot){
      initKeywords();

      const char *c = cRoot;

      strNode *nodeRoot = new strNode();
      strNode *nodePos  = nodeRoot;

      int status = readingCode;

      int lineCount = 0;

      while(*c != '\0'){
        const char *cEnd = readLine(c);

        std::string line = strip(c, cEnd - c);

        if(line.size()){
          if(status != insideCommentBlock){
            status = stripComments(line);
            strip(line);

            if(line.size()){
              nodePos->originalLine = lineCount;
              nodePos = nodePos->push(line);
            }
          }
          else{
            status = stripComments(line);
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

    strNode* labelCode(strNode *lineNodeRoot){
      initKeywords();

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
            const int delimeterChars = isAWordDelimeter(cLeft);

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

            nodePos->type  = presetValue;
            nodePos->depth = depth;

            cLeft = cRight;
          }
          else if(loadNumber){
            const int delimeterChars = isAWordDelimeter(cLeft);

            skipNumber(cRight);

            if(!firstSectionNode){
              nodePos = nodePos->push( std::string(cLeft, (cRight - cLeft)) );
            }
            else{
              nodePos = nodePos->pushDown( std::string(cLeft, (cRight - cLeft)) );
              firstSectionNode = false;
            }

            nodePos->type  = presetValue;
            nodePos->depth = depth;

            cLeft = cRight;
          }
          else{
            const int delimeterChars = isAWordDelimeter(cLeft);

            if(delimeterChars){
              strNode *newNode = new strNode(std::string(cLeft, delimeterChars));

              newNode->type  = keywordType[newNode->value];
              newNode->depth = depth;

              if(newNode->type & startSection){
                if(!firstSectionNode)
                  nodePos = nodePos->push(newNode);
                else
                  nodePos = nodePos->pushDown(newNode);

                ++depth;

                firstSectionNode = true;
              }
              else if(newNode->type & endSection){
                if(!firstSectionNode)
                  nodePos = nodePos->up;

                delete newNode;

                --depth;

                firstSectionNode = false;
              }
              else if(newNode->type & macroKeywordType){
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
              skipWord(cRight);

              if(!firstSectionNode){
                nodePos = nodePos->push( std::string(cLeft, (cRight - cLeft)) );
              }
              else{
                nodePos = nodePos->pushDown( std::string(cLeft, (cRight - cLeft)) );
                firstSectionNode = false;
              }

              keywordTypeMapIterator it = keywordType.find(nodePos->value);

              if(it == keywordType.end())
                nodePos->type = unknownVariable;
              else{
                nodePos->type = it->second;

                // Merge [else] [if] -> [else if]
                if((nodePos->type & flowControlType)       &&
                   (nodePos->left)                         &&
                   (nodePos->left->type & flowControlType) &&
                   (nodePos->value == "if")                &&
                   (nodePos->left->value == "else")){

                  nodePos->value = "else if";

                  strNode *elseNode = nodePos->left;

                  nodePos->left        = nodePos->left->left;
                  nodePos->left->right = nodePos;

                  delete elseNode->pop();
                }
                else if((nodePos->type & specialKeywordType) &&
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

              nodePos->depth = depth;

              cLeft = cRight;
            }
          }
        }

        lineNodePos = lineNodePos->right;
      }

      if(nodePos != nodeRoot)
        popAndGoRight(nodeRoot);

      free(lineNodeRoot);

      return nodeRoot;
    }

    void initKeywords(){
      if(keywordsAreInitialized)
        return;

      keywordsAreInitialized = true;

      //---[ Operator Info ]--------------
      keywordType["!"]  = lUnitaryOperatorType;
      keywordType["%"]  = binaryOperatorType;
      keywordType["&"]  = (binaryOperatorType | qualifierType);
      keywordType["("]  = startParentheses;
      keywordType[")"]  = endParentheses;
      keywordType["*"]  = (binaryOperatorType | qualifierType);
      keywordType["+"]  = (lUnitaryOperatorType | binaryOperatorType);
      keywordType[","]  = binaryOperatorType;
      keywordType["-"]  = (lUnitaryOperatorType | binaryOperatorType);
      keywordType["."]  = binaryOperatorType;
      keywordType["/"]  = binaryOperatorType;
      keywordType[":"]  = endStatement;
      keywordType[";"]  = endStatement;
      keywordType["<"]  = binaryOperatorType;
      keywordType["="]  = binaryOperatorType;
      keywordType[">"]  = binaryOperatorType;
      keywordType["?"]  = ternaryOperatorType;
      keywordType["["]  = startBracket;
      keywordType["]"]  = endBracket;
      keywordType["^"]  = (binaryOperatorType | qualifierType);
      keywordType["{"]  = startBrace;
      keywordType["|"]  = binaryOperatorType;
      keywordType["}"]  = endBrace;
      keywordType["~"]  = lUnitaryOperatorType;
      keywordType["!="] = assOperatorType;
      keywordType["%="] = assOperatorType;
      keywordType["&&"] = binaryOperatorType;
      keywordType["&="] = assOperatorType;
      keywordType["*="] = assOperatorType;
      keywordType["+="] = assOperatorType;
      keywordType["++"] = unitaryOperatorType;
      keywordType["-="] = assOperatorType;
      keywordType["--"] = unitaryOperatorType;
      keywordType["->"] = binaryOperatorType;
      keywordType["/="] = assOperatorType;
      keywordType["::"] = binaryOperatorType;
      keywordType["<<"] = binaryOperatorType;
      keywordType["<="] = binaryOperatorType;
      keywordType["=="] = binaryOperatorType;
      keywordType[">="] = binaryOperatorType;
      keywordType[">>"] = binaryOperatorType;
      keywordType["^="] = assOperatorType;
      keywordType["|="] = assOperatorType;
      keywordType["||"] = binaryOperatorType;

      keywordType["#"] = macroKeywordType;

      //---[ Types & Specifiers ]---------
      std::string suffix[6] = {"", "2", "3", "4", "8", "16"};

      for(int i = 0; i < 6; ++i){
        keywordType[std::string("int")    + suffix[i]] = specifierType;
        keywordType[std::string("bool")   + suffix[i]] = specifierType;
        keywordType[std::string("char")   + suffix[i]] = specifierType;
        keywordType[std::string("long")   + suffix[i]] = specifierType;
        keywordType[std::string("short")  + suffix[i]] = specifierType;
        keywordType[std::string("float")  + suffix[i]] = specifierType;
        keywordType[std::string("double") + suffix[i]] = specifierType;
      }

      keywordType["void"]          = specifierType;
      keywordType["__attribute__"] = specifierType; // [--]

      keywordType["long"]     = (qualifierType | specifierType);
      keywordType["short"]    = (qualifierType | specifierType);
      keywordType["signed"]   = qualifierType;
      keywordType["unsigned"] = qualifierType;

      keywordType["inline"] = qualifierType;
      keywordType["static"] = qualifierType;
      keywordType["extern"] = qualifierType;

      keywordType["const"]    = (qualifierType | occaKeywordType);
      keywordType["restrict"] = (qualifierType | occaKeywordType);
      keywordType["volatile"] = (qualifierType | occaKeywordType);
      keywordType["aligned"]  = (qualifierType | occaKeywordType);
      keywordType["register"] = qualifierType;

      keywordType["occaConst"]    = (qualifierType | occaKeywordType);
      keywordType["occaRestrict"] = (qualifierType | occaKeywordType);
      keywordType["occaVolatile"] = (qualifierType | occaKeywordType);
      keywordType["occaAligned"]  = (qualifierType | occaKeywordType);
      keywordType["occaConstant"] = (qualifierType | occaKeywordType);

      keywordType["class"]   = (structType);
      keywordType["enum"]    = (structType | qualifierType);
      keywordType["union"]   = (structType | qualifierType);
      keywordType["struct"]  = (structType | qualifierType);
      keywordType["typedef"] = (typedefType | qualifierType);

      //---[ Non-standard ]-------------
      keywordType["__attribute__"] = (qualifierType | specialKeywordType);

      //---[ C++ ]----------------------
      keywordType["virtual"]   = qualifierType;

      keywordType["namespace"] = (specifierType | structType);

      //---[ Constants ]------------------
      keywordType["..."]   = presetValue;
      keywordType["true"]  = presetValue;
      keywordType["false"] = presetValue;

      //---[ Flow Control ]---------------
      keywordType["if"]   = flowControlType;
      keywordType["else"] = flowControlType;

      keywordType["for"] = flowControlType;

      keywordType["do"]    = flowControlType;
      keywordType["while"] = flowControlType;

      keywordType["switch"]  = flowControlType;
      keywordType["case"]    = specialKeywordType;
      keywordType["default"] = specialKeywordType;

      keywordType["break"]    = specialKeywordType;
      keywordType["continue"] = specialKeywordType;
      keywordType["return"]   = specialKeywordType;
      keywordType["goto"]     = specialKeywordType;

      //---[ OCCA Keywords ]--------------
      keywordType["kernel"]    = (qualifierType | occaKeywordType);
      keywordType["texture"]   = (qualifierType | occaKeywordType);
      keywordType["shared"]    = (qualifierType | occaKeywordType);
      keywordType["exclusive"] = (qualifierType | occaKeywordType);

      keywordType["occaKernel"]   = (qualifierType | occaKeywordType);
      keywordType["occaFunction"] = (qualifierType | occaKeywordType);
      keywordType["occaDeviceFunction"] = (qualifierType | occaKeywordType);
      keywordType["occaPointer"]  = (qualifierType | occaKeywordType);
      keywordType["occaVariable"] = (qualifierType | occaKeywordType);
      keywordType["occaShared"]   = (qualifierType | occaKeywordType);

      keywordType["occaKernelInfoArg"] = (presetValue | occaKeywordType);
      keywordType["occaKernelInfo"]    = (presetValue | occaKeywordType);

      keywordType["occaPrivate"]      = (presetValue | occaKeywordType);
      keywordType["occaPrivateArray"] = (presetValue | occaKeywordType);

      keywordType["barrier"]        = (presetValue | occaKeywordType);
      keywordType["localMemFence"]  = (presetValue | occaKeywordType);
      keywordType["globalMemFence"] = (presetValue | occaKeywordType);

      keywordType["occaBarrier"]        = (presetValue | occaKeywordType);
      keywordType["occaLocalMemFence"]  = (presetValue | occaKeywordType);
      keywordType["occaGlobalMemFence"] = (presetValue | occaKeywordType);

      keywordType["occaInnerFor0"] = (forStatementType | occaStatementType);
      keywordType["occaInnerFor1"] = (forStatementType | occaStatementType);
      keywordType["occaInnerFor2"] = (forStatementType | occaStatementType);

      keywordType["occaOuterFor0"] = (forStatementType | occaStatementType);
      keywordType["occaOuterFor1"] = (forStatementType | occaStatementType);
      keywordType["occaOuterFor2"] = (forStatementType | occaStatementType);

      keywordType["occaInnerId0"] = (presetValue | occaKeywordType);
      keywordType["occaInnerId1"] = (presetValue | occaKeywordType);
      keywordType["occaInnerId2"] = (presetValue | occaKeywordType);

      keywordType["occaOuterId0"] = (presetValue | occaKeywordType);
      keywordType["occaOuterId1"] = (presetValue | occaKeywordType);
      keywordType["occaOuterId2"] = (presetValue | occaKeywordType);

      keywordType["occaGlobalId0"] = (presetValue | occaKeywordType);
      keywordType["occaGlobalId1"] = (presetValue | occaKeywordType);
      keywordType["occaGlobalId2"] = (presetValue | occaKeywordType);

      keywordType["occaInnerDim0"] = (presetValue | occaKeywordType);
      keywordType["occaInnerDim1"] = (presetValue | occaKeywordType);
      keywordType["occaInnerDim2"] = (presetValue | occaKeywordType);

      keywordType["occaOuterDim0"] = (presetValue | occaKeywordType);
      keywordType["occaOuterDim1"] = (presetValue | occaKeywordType);
      keywordType["occaOuterDim2"] = (presetValue | occaKeywordType);

      keywordType["occaGlobalDim0"] = (presetValue | occaKeywordType);
      keywordType["occaGlobalDim1"] = (presetValue | occaKeywordType);
      keywordType["occaGlobalDim2"] = (presetValue | occaKeywordType);

      //---[ CUDA Keywords ]--------------
      keywordType["threadIdx"] = (unknownVariable | cudaKeywordType);
      keywordType["blockDim"]  = (unknownVariable | cudaKeywordType);
      keywordType["blockIdx"]  = (unknownVariable | cudaKeywordType);
      keywordType["gridDim"]   = (unknownVariable | cudaKeywordType);

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

        keywordType["occa"       + cmf] = presetValue;
        keywordType["occaFast"   + cmf] = presetValue;
        keywordType["occaNative" + cmf] = presetValue;
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
        keywordType["using"] = ;
        keywordType["namespace"] = ;
        keywordType["template"] = ;
        ================================*/
    }
  };
};
