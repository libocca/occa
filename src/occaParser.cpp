#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
    intVector_t loadedLanguageVec;

    int loadedLanguage(){
      return loadedLanguageVec.back();
    }

    void pushLanguage(const int language){
      loadedLanguageVec.push_back(language);
      loadKeywords(language);
    }

    int popLanguage(){
      const int ret = loadedLanguageVec.back();

      loadedLanguageVec.pop_back();
      loadKeywords(loadedLanguageVec.back());

      return ret;
    }

    parserBase::parserBase(){
      env::initialize();

      parsingLanguage = parserInfo::parsingC;

      macrosAreInitialized = false;
      globalScope = new statement(*this);

      warnForMissingBarriers     = true;
      warnForBarrierConditionals = true;
      magicEnabled               = false;
    }

    const std::string parserBase::parseFile(const std::string &filename_,
                                            const int parsingLanguage_){

      filename = filename_;

      parsingLanguage = parsingLanguage_;
      pushLanguage(parsingLanguage);

      const char *cRoot = cReadFile(filename);

      const std::string parsedContent = parseSource(cRoot);

      delete [] cRoot;

      return parsedContent;
    }

    const std::string parserBase::parseSource(const char *cRoot){
      expNode allExp = splitAndPreprocessContent(cRoot, parsingLanguage);
      // allExp.print();
      // throw 1;

      loadLanguageTypes();

      globalScope->loadAllFromNode(allExp, parsingLanguage);
      // std::cout << (std::string) *globalScope;
      // throw 1;

      applyToAllStatements(*globalScope, &parserBase::splitTileOccaFors);
      // std::cout << (std::string) *globalScope;
      // throw 1;

      markKernelFunctions(*globalScope);
      labelNativeKernels();

      if(magicEnabled){
        magician::castMagicOn(*this);
        std::cout << (std::string) *globalScope;
        throw 1;
      }

      applyToAllStatements(*globalScope, &parserBase::setupCudaVariables);
      applyToAllStatements(*globalScope, &parserBase::setupOccaVariables);

      checkOccaBarriers(*globalScope);
      addOccaBarriers();

      addFunctionPrototypes();
      updateConstToConstant();

      addOccaFors();

      applyToAllStatements(*globalScope, &parserBase::setupOccaFors);

      applyToAllKernels(*globalScope, &parserBase::floatSharedAndExclusivesUp);

      // [-] Missing
      modifyTextureVariables();

      addArgQualifiers();

      loadKernelInfos();

      applyToAllStatements(*globalScope, &parserBase::modifyExclusiveVariables);

      return (std::string) *globalScope;
    }

    //---[ Macro Parser Functions ]-------
    std::string parserBase::getMacroName(const char *&c){
      const char *cStart = c;
      skipWord(cStart);
      skipWhitespace(cStart);
      c = cStart;

      while((*c != '\0') &&
            (*c != '(')  &&
            !isWhitespace(*c)){

        if((c[0] == '#') && (c[1] == '<')){
          while((c[0] != '\0') &&
                ((c[0] != '#') || (c[1] != '>'))){

            ++c;
          }

          c += 2;
        }
        else
          ++c;
      }

      std::string name(cStart, c - cStart);

      if(macroMap.find(name) == macroMap.end())
        applyMacros(name);

      return name;
    }

    std::string parserBase::getMacroIncludeFile(const char *&c){
      const char *cStart = c;
      skipWord(cStart);

      c = cStart;

      while((*c != '\n') &&
            (*c != '\0')){

        ++c;
      }

      std::string iFilename = strip(cStart, c - cStart);
      const int chars = iFilename.size();

      const bool skipFirst = ((iFilename[0] == '"') ||
                              (iFilename[0] == '<'));

      const bool skipLast  = ((iFilename[chars - 1] == '"') ||
                              (iFilename[chars - 1] == '<'));

      return iFilename.substr(skipFirst, chars - (skipFirst + skipLast));
    }

    typeHolder parserBase::evaluateMacroStatement(const char *c){
      return evaluateString(c, this);
    }

    bool parserBase::evaluateMacroBoolStatement(const char *c){
      typeHolder th = evaluateMacroStatement(c);

      return th.to<bool>();
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
        const char *cStart = c;

        skipTo(c, ",)");

        macroArgMap[strip(cStart, c - cStart)] = (info.argc++);

        if(*c == ')')
          break;

        ++c;
      }

      ++c; // ')'

      if(isWhitespace(*c)){
        info.parts[partPos] += ' ';
        skipWhitespace(c);
      }

      while(*c != '\0'){
        const char *cStart = c;

        if(isAString(c)){
          skipString(c, parsingLanguage);

          info.parts[partPos] += std::string(cStart, (c - cStart));
          continue;
        }

        const int delimiterChars = skipWord(c);

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
        c += delimiterChars;

        if(cStart != c)
          info.parts[partPos] += std::string(cStart, c - cStart);

        if(isWhitespace(*c)){
          info.parts[partPos] += ' ';
          skipWhitespace(c);
        }
      }
    }

    int parserBase::loadMacro(expNode &allExp, int leafPos, const int state){
      return loadMacro(allExp, leafPos, allExp[leafPos].value, state);
    }

    int parserBase::loadMacro(const std::string &line, const int state){
      expNode dummyExpRoot;

      return loadMacro(dummyExpRoot, -1, line, state);
    }

    int parserBase::loadMacro(expNode &allExp, int leafPos,
                              const std::string &line, const int state){

      const char *c = (line.c_str() + 1); // line[0] = #

      while(*c != '\0'){
        skipWhitespace(c);
        const char *cEnd = c;
        skipToWhitespace(cEnd);

        if(stringsAreEqual(c, (cEnd - c), "if")){
          c = cEnd;

          bool isTrue = evaluateMacroBoolStatement(c);

          if(isTrue)
            return (startHash | readUntilNextHash);
          else
            return (startHash | ignoreUntilNextHash);
        }

        else if(stringsAreEqual(c, (cEnd - c), "elif")){
          if((state & readUntilNextHash) || (state & ignoreUntilEnd))
            return ignoreUntilEnd;

          c = cEnd;

          bool isTrue = evaluateMacroBoolStatement(c);

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

          return (state);
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

          // Nothing to edit, just keep the #include for the compiler
          if(leafPos == -1)
            return (state | keepMacro);

          std::string includeFile = getMacroIncludeFile(c);

          if(includeFile[0] == '~'){
            includeFile = (env::HOME +
                           includeFile.substr(1, includeFile.size() - 1));
          }
          else if(includeFile[0] != '/'){
            includeFile = findFileInPath(includeFile);
          }
          else{
            std::string prefix = getFilePrefix(filename);

            if(prefix.size())
              includeFile = findFileInPath(prefix + '/' + includeFile);
          }

          if(includeFile == "")
            return (state | forceLineRemoval);

          const char *cRoot = cReadFile(includeFile);

          expNode includeExpRoot = splitContent(cRoot, parsingLanguage);

          delete [] cRoot;

          // Empty include file
          if(includeExpRoot.leafCount == 0)
            return (state | forceLineRemoval);

          leafPos = allExp.insertExpAt(includeExpRoot, leafPos + 1);

          delete includeExpRoot.leaves;

          return (state | forceLineRemoval);
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
          skipString(c, parsingLanguage);

          newLine += std::string(cStart, (c - cStart));
          continue;
        }

        int delimiterChars = skipWord(c);

        std::string word = std::string(cStart, c - cStart);

        macroMapIterator it = macroMap.find(word);

        while(delimiterChars == 2){
          //---[ #< #> ]----------------
          if(stringsAreEqual(c, delimiterChars, "#<")){
            c += 2;
            cStart = c;

            while((c[0] != '\0') &&
                  ((c[0] != '#') || (c[1] != '>'))){

              ++c;
            }

            std::string expr(cStart, c - cStart);
            const char *c_expr = expr.c_str();

            std::string expr2 = (std::string) evaluateMacroStatement(c_expr);

            while(expr != expr2){
              expr = expr2;
              applyMacros(expr2);
            }

            word += expr;
            applyMacros(word);

            // Don't include delimiter chars
            c += 2;
            delimiterChars = 0;
          }
          //---[ ## ]-------------------
          else if(stringsAreEqual(c, delimiterChars, "##")){
            c += 2;

            cStart = c;
            delimiterChars = skipWord(c);

            std::string word2 = std::string(cStart, c - cStart);

            it = macroMap.find(word2);

            if(it != macroMap.end()){
              macroInfo &info = macros[it->second];
              word += info.parts[0];
            }
            else
              word += word2;
          }
          else
            break;
          //============================
        }

        it = macroMap.find(word);

        if(it != macroMap.end()){
          foundMacro = true;

          macroInfo &info = macros[it->second];

          if(!info.isAFunction || (*c != '(')){
            newLine += info.parts[0];
          }
          else{
            std::vector<std::string> args;

            cStart = c + 1;
            skipWhitespace(cStart);

            skipPair(c);

            const char *cEnd = c;
            c = cStart;

            OCCA_CHECK(*c != '\0',
                       "Missing ')' in ["
                       << info.name
                       << "(" << std::string(cStart, cEnd - cStart - 1) << ")]");

            while(c < cEnd){
              if(*c == ','){
                args.push_back( strip(cStart, c - cStart) );
                cStart = ++c; // Skip the [,]
              }
              else{
                if(segmentPair(*c))
                  skipPair(c);
                else if(isAString(c))
                  skipString(c);
                else
                  ++c;
              }
            }

            c = cEnd;

            if(cStart < (cEnd - 1))
              args.push_back( std::string(cStart, cEnd - cStart - 1) );

            newLine += info.applyArgs(args);
          }
        }
        else{
          newLine += word;
        }

        cStart = c;
        c += delimiterChars;

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

    void parserBase::preprocessMacros(expNode &allExp){
      std::stack<int> statusStack;
      std::vector<int> linesIgnored;

      int currentState = doNothing;

      for(int linePos = 0; linePos < (allExp.leafCount); ++linePos){
        std::string &line = allExp[linePos].value;
        bool ignoreLine = false;

        if(line[0] == '#'){
          const int oldState = currentState;

          currentState = loadMacro(allExp, linePos, currentState);

          if(currentState & keepMacro){
            currentState &= ~keepMacro;
          }
          else if(currentState & forceLineRemoval){
            currentState &= ~forceLineRemoval;
            ignoreLine = true;
          }
          else
            ignoreLine = false; // Keep macros for now

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

        if(ignoreLine)
          linesIgnored.push_back(linePos);
      }

      if(linesIgnored.size() == 0)
        return;

      if(linesIgnored.back() != (allExp.leafCount - 1))
        linesIgnored.push_back(allExp.leafCount - 1);

      const int ignoreCount = (int) linesIgnored.size();

      int start = (linesIgnored[0] + 1);
      int pos   = linesIgnored[0];

      for(int linePos = 1; linePos < ignoreCount; ++linePos){
        int end = linesIgnored[linePos];

        for(int i = start; i < end; ++i)
          allExp[pos++].value = allExp[i].value;

        start = (end + 1);
      }

      allExp.leafCount = pos;
    }

    expNode parserBase::splitAndPreprocessContent(const std::string &s,
                                                  const int parsingLanguage){
      return splitAndPreprocessContent(s.c_str(),
                                       parsingLanguage);
    }

    expNode parserBase::splitAndPreprocessContent(const char *cRoot,
                                                  const int parsingLanguage){
      expNode allExp;

      pushLanguage(parsingLanguage);

      allExp = splitContent(cRoot, parsingLanguage);

      initMacros(parsingLanguage);
      preprocessMacros(allExp);

      labelCode(allExp, parsingLanguage);

      allExp.setNestedSInfo(globalScope);

      popLanguage();

      return allExp;
    }
    //====================================

    void parserBase::initMacros(const int parsingLanguage){
      if(parsingLanguage & parserInfo::parsingFortran)
        initFortranMacros();

      if(macrosAreInitialized)
        return;

      macrosAreInitialized = true;

      //---[ Macros ]---------------------
      loadMacro("#define kernel occaKernel");

      loadMacro("#define barrier        occaBarrier");
      loadMacro("#define localMemFence  occaLocalMemFence");
      loadMacro("#define globalMemFence occaGlobalMemFence");

      loadMacro("#define atomicAdd  occaAtomicAdd");
      loadMacro("#define atomicSub  occaAtomicSub");
      loadMacro("#define atomicSwap occaAtomicSwap");
      loadMacro("#define atomicInc  occaAtomicInc");
      loadMacro("#define atomicDec  occaAtomicDec");
      loadMacro("#define atomicMin  occaAtomicMin");
      loadMacro("#define atomicMax  occaAtomicMax");
      loadMacro("#define atomicAnd  occaAtomicAnd");
      loadMacro("#define atomicOr   occaAtomicOr");
      loadMacro("#define atomicXor  occaAtomicXor");

      loadMacro("#define atomicAdd64  occaAtomicAdd64");
      loadMacro("#define atomicSub64  occaAtomicSub64");
      loadMacro("#define atomicSwap64 occaAtomicSwap64");
      loadMacro("#define atomicInc64  occaAtomicInc64");
      loadMacro("#define atomicDec64  occaAtomicDec64");

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
      pushLanguage(parserInfo::parsingC);

      int parts[6]            = {1, 2, 3, 4, 8, 16};
      std::string suffix[6]   = {"", "2", "3", "4", "8", "16"};
      std::string baseType[7] = {"int"  ,
                                 "bool" ,
                                 "char" ,
                                 "long" ,
                                 "short",
                                 "float",
                                 "double"};

      std::stringstream ss;

      for(int t = 0; t < 7; ++t){
        for(int n = 0; n < 6; ++n){
          typeInfo &type = *(new typeInfo);

          if(n == 0){
            type.name     = baseType[t] + suffix[n];
            type.baseType = &type;

            globalScope->scopeTypeMap[type.name] = &type;
          }
          else{
            ss << "struct " << baseType[t] << parts[n] << " {\n";

            for(int n2 = 0; n2 < parts[n]; ++n2){
              const char varLetter = ('w' + ((n2 + 1) % 4));
              const char varNum1   = ((n2 < 10) ? ('0' + n2) : ('a' + (n2 - 10)));
              const char varNum2   = ((n2 < 10) ? ('0' + n2) : ('A' + (n2 - 10)));

              const bool needsUnion = ((n2 < 4) || (10 <= n2));

              std::string tab = (needsUnion ? "    " : "  ");

              if(needsUnion)
                ss << "  union {\n";

              if(n2 < 4)
                ss << tab << baseType[t] << " " << varLetter << ";\n";

              ss << tab << baseType[t] << " s" << varNum1 << ";\n";

              if(10 <= n2)
                ss << tab << baseType[t] << " s" << varNum2 << ";\n";

              if(needsUnion)
                ss << "  };\n";
            }

            ss << "};";

            expNode typeExp = globalScope->createPlainExpNodeFrom(ss.str());

            type.loadFrom(*globalScope, typeExp);
            // typeExp.free(); [<>] Errors out

            globalScope->scopeTypeMap[type.name] = &type;

            ss.str("");
          }
        }
      }

      globalScope->addTypedef("void");
      globalScope->addTypedef("__builtin_va_list");

      popLanguage();
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

    void parserBase::applyToAllKernels(statement &s,
                                       applyToAllStatements_t func){
      if(statementIsAKernel(s)){
        (this->*func)(s);
        return;
      }

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
      if(s.info & smntType::functionStatement){
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
      statement *ret = ((s.info == smntType::occaFor) ? &s : NULL);

      statement *sUp = &s;

      while(sUp){
        if(sUp->info == smntType::occaFor)
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
      if( !(s.info & smntType::forStatement) ||
          (s.getForStatementCount() <= 3) ){

        return;
      }

      statement *spKernel = getStatementKernel(s);

      if(spKernel == NULL)
        return;

      if(statementKernelUsesNativeOCCA(s))
        return;

      occaLoopInfo loopInfo(s, parsingLanguage);

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

      s.info = smntType::occaFor;
    }

    bool parserBase::statementIsOccaOuterFor(statement &s){
      if(s.info == smntType::occaFor){
        std::string &forName = s.expRoot.value;

        if((forName.find("occaOuterFor") != std::string::npos) &&
           ((forName == "occaOuterFor0") ||
            (forName == "occaOuterFor1") ||
            (forName == "occaOuterFor2"))){

          return true;
        }
      }

      return false;
    }

    bool parserBase::statementIsOccaInnerFor(statement &s){
      if(s.info == smntType::occaFor){
        std::string &forName = s.expRoot.value;

        if((forName.find("occaInnerFor") != std::string::npos) &&
           ((forName == "occaInnerFor0") ||
            (forName == "occaInnerFor1") ||
            (forName == "occaInnerFor2"))){

          return true;
        }
      }

      return false;
    }

    bool parserBase::statementHasOccaOuterFor(statement &s){
      if(statementIsOccaOuterFor(s))
        return true;

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        if( statementHasOccaOuterFor(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    bool parserBase::statementHasOccaFor(statement &s){
      if(s.info == smntType::occaFor)
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
      if((s.info == smntType::forStatement) &&
         (s.getForStatementCount() == 4)){

        return true;
      }

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

    void parserBase::splitTileOccaFors(statement &s){
      if((s.info != smntType::forStatement) ||
         (s.getForStatementCount() < 4)){

        return;
      }

      expNode &tagNode = *(s.getForStatement(3));

      if((tagNode[0].value != "(")   ||
         (tagNode[0].leafCount != 2) ||
         (tagNode[0][0].value != "tile")){

        return;
      }

      expNode &initNode   = *(s.getForStatement(0));
      expNode &checkNode  = *(s.getForStatement(1));
      expNode &updateNode = *(s.getForStatement(2));

      expNode &csvCheckNode  = *(checkNode.makeCsvFlatHandle());
      expNode &csvUpdateNode = *(updateNode.makeCsvFlatHandle());

      expNode &csvTileDims = *(tagNode[0][1].makeCsvFlatHandle());

      //---[ Checks ]---------------------------
      //  ---[ Tile Dim ]-------------
      const int tileDim = csvTileDims.leafCount;

      OCCA_CHECK((1 <= tileDim) && (tileDim <= 3),
                 "Only 1D, 2D, and 3D tiling are supported:\n" << s.onlyThisToString());

      int varsInInit = ((initNode.info & expType::declaration) ?
                        initNode.getVariableCount()            :
                        initNode.getUpdatedVariableCount());

      OCCA_CHECK(varsInInit == 1,
                 "Only one iterator can be initialized:\n" << s.onlyThisToString());

      expNode *varInitNode = ((initNode.info & expType::declaration) ?
                              initNode.getVariableInitNode(0)        :
                              initNode.getUpdatedVariableSetNode(0));

      expNode *csvInitValueNode_;

      if(tileDim == 1){
        csvInitValueNode_ = varInitNode->makeCsvFlatHandle();
      }
      else {
        OCCA_CHECK(varInitNode->value == "{",
                   "Iterator is not defined properly (e.g. int2 i = {0,0}):\n" << s.onlyThisToString());

        csvInitValueNode_ = varInitNode->leaves[0]->makeCsvFlatHandle();
      }

      expNode &csvInitValueNode = *csvInitValueNode_;

      //  ---[ Proper init var ]------
      const bool varIsDeclared = (initNode.info & expType::declaration);

      varInfo &var = ((initNode.info & expType::declaration)        ?
                      initNode.getVariableInfoNode(0)->getVarInfo() :
                      initNode.getUpdatedVariableInfoNode(0)->getVarInfo());

      std::string &varTypeN = var.baseType->baseType->name;
      std::string varType, suffix;

      if(1 < tileDim){
        suffix = '0';
        suffix[0] += tileDim;
      }

      if(     varTypeN == ("int"   + suffix)) varType = "int";
      else if(varTypeN == ("char"  + suffix)) varType = "char";
      else if(varTypeN == ("long"  + suffix)) varType = "long";
      else if(varTypeN == ("short" + suffix)) varType = "short";

      OCCA_CHECK(0 < varType.size(),
                 "Iterator [" << var << "] is not a proper type (e.g. int" << suffix << ')');

      //  ---[ Proper check vars ]----
      int varsInCheck = csvCheckNode.leafCount;

      OCCA_CHECK(varsInCheck == tileDim,
                 "Only one variable can be checked:\n" << s.onlyThisToString());

      expNode **orderBuffer = new expNode*[csvCheckNode.leafCount];
      bool *checkIterOnLeft = new bool[csvCheckNode.leafCount];

      for(int dim = 0; dim < tileDim; ++dim)
        orderBuffer[dim] = NULL;

      for(int dim = 0; dim < tileDim; ++dim){
        expNode &check = csvCheckNode[dim];
        int dim2 = dim;

        OCCA_CHECK((check.info == expType::LR) &&
                   ((check.value == "<=") ||
                    (check.value == "<" ) ||
                    (check.value == ">" ) ||
                    (check.value == ">=")),
                   "Error on: " << s.onlyThisToString() << "\n\n"
                   << "Check operator must be in [<=, <, >, >=]: " << check.toString());

        int side;

        for(side = 0; side < 2; ++side){
          if(tileDim == 1){
            if((check[side].value == var.name)){
              checkIterOnLeft[dim2] = (side == 0);

              break;
            }
          }
          else {
            if((check[side].value    == ".") &&
               (check[side][0].value == var.name)){

              dim2 = (check[side][1].value[0] - 'x');
              checkIterOnLeft[dim2] = (side == 0);

              break;
            }
          }
        }

        OCCA_CHECK(side < 2,
                   "Error on: " << s.onlyThisToString() << "\n\n"
                   << "Variable checks must look like:\n"
                   "  X op Y where op can be [<=, <, >, >=]\n"
                   "  X or Y must be for-loop iterator\n"
                   "  For 2D or 3D tiling: X.x < Y, X.y < Y, X.z < Y (order doesn't matter)");

        orderBuffer[dim2] = &(csvCheckNode[dim]);
      }

      for(int dim = 0; dim < tileDim; ++dim){
        OCCA_CHECK(orderBuffer[dim] != NULL,
                   var.name << '.' << (char) ('x' + dim) << " needs to be checked: " << s.onlyThisToString());

        csvCheckNode.leaves[dim] = orderBuffer[dim];
        orderBuffer[dim]         = NULL;
      }

      //  ---[ Proper update vars ]---
      int varsInUpdate = csvUpdateNode.leafCount;

      OCCA_CHECK(varsInUpdate == tileDim,
                 "Only one variable can be updated:\n" << s.onlyThisToString());

      for(int dim = 0; dim < tileDim; ++dim){
        expNode &update = csvUpdateNode[dim];
        int dim2 = dim;

        OCCA_CHECK((update.value == "++") ||
                   (update.value == "--") ||
                   (update.value == "+=") ||
                   (update.value == "-="),
                   "Update operator must be in [++, --, +=, -=]: " << update.toString());

        if(1 < tileDim){
          OCCA_CHECK(update[0][0].value == var.name,
                     "Iterator [" << var.name << "] is not updated, [" << update[0][0].value << "] is updated instead");

          dim2 = (update[0][1].value[0] - 'x');
        }

        orderBuffer[dim2] = &(csvUpdateNode[dim]);
      }

      for(int dim = 0; dim < tileDim; ++dim){
        OCCA_CHECK(orderBuffer[dim] != NULL,
                   var.name << '.' << (char) ('x' + dim) << " needs to be updated: " << s.onlyThisToString());

        csvUpdateNode.leaves[dim] = orderBuffer[dim];
      }

      delete [] orderBuffer;
      //========================================

      // Placeholders for outer and inner for-loops
      statement **oStatements = new statement*[tileDim];
      statement **iStatements = new statement*[tileDim];

      // Swap s's statementNode with outer-most for-loop
      oStatements[tileDim - 1] = s.up->makeSubStatement();
      s.getStatementNode()->value = oStatements[tileDim - 1];

      for(int dim = (tileDim - 2); 0 <= dim; --dim){
        oStatements[dim] = oStatements[dim + 1]->makeSubStatement();
        oStatements[dim + 1]->addStatement(oStatements[dim]);
      }

      iStatements[tileDim - 1] = oStatements[0]->makeSubStatement();
      oStatements[0]->addStatement(iStatements[tileDim - 1]);

      for(int dim = (tileDim - 2); 0 <= dim; --dim){
        iStatements[dim] = iStatements[dim + 1]->makeSubStatement();
        iStatements[dim + 1]->addStatement(iStatements[dim]);
      }

      // Place s's statementNode's in inner-most for-loop
      iStatements[0]->statementCount = s.statementCount;
      iStatements[0]->statementStart = s.statementStart;
      iStatements[0]->statementEnd   = s.statementEnd;

      statementNode *sn = iStatements[0]->statementStart;

      while(sn){
        sn->value->up = iStatements[0];
        sn = sn->right;
      }

      std::stringstream ss;

      for(int dim = 0; dim < tileDim; ++dim){
        statement &os = *(oStatements[dim]);
        statement &is = *(iStatements[dim]);

        os.info = smntType::forStatement;
        is.info = smntType::forStatement;

        expNode &check  = csvCheckNode[dim];
        expNode &update = csvUpdateNode[dim];

        std::string oTileVar = "__occa_oTileVar0";
        oTileVar[oTileVar.size() - 1] += dim;

        std::string iTileVar = "__occa_iTileVar0";
        iTileVar[iTileVar.size() - 1] += dim;

        ss << "for("
           << varType << ' ' << oTileVar << " = " << csvInitValueNode[dim].toString() << "; ";

        if(checkIterOnLeft[dim])
          ss << oTileVar << check.value << check[1].toString() << "; ";
        else
          ss << check[0].toString() << check.value << oTileVar << "; ";

        if(update.info != expType::LR){
          if(update.value == "++")
            ss << oTileVar << " += " << csvTileDims[dim] << "; ";
          else
            ss << oTileVar << " -= " << csvTileDims[dim] << "; ";
        }
        else {
          ss << oTileVar << update.value << csvTileDims[dim] << "; ";
        }

        ss << "outer" << dim << ')';

        std::string outerForSource = ss.str();

        ss.str("");

        std::string varName = var.name;

        if(1 < tileDim)
          varName = iTileVar;

        ss << "for(";

        if(1 < tileDim)
          ss << varType << ' ';

        ss << varName << " = " << oTileVar << "; ";

        if(checkIterOnLeft[dim])
          ss << varName << check.value << '(' << oTileVar << " + " << csvTileDims[dim] << "); ";
        else
          ss << '(' << oTileVar << " + " << csvTileDims[dim] << ')' << check.value << varName << "; ";

        csvUpdateNode[dim][0].free();
        csvUpdateNode[dim][0].info  = expType::printValue;
        csvUpdateNode[dim][0].value = varName;

        ss << csvUpdateNode[dim].toString() << "; ";

        ss << "inner" << dim << ')';

        std::string innerForSource = ss.str();

        ss.str("");

        expNode outerExp = s.createOrganizedExpNodeFrom(outerForSource);
        expNode innerExp = s.createOrganizedExpNodeFrom(innerForSource);

        expNode::swap(os.expRoot, outerExp);
        expNode::swap(is.expRoot, innerExp);

        outerExp.free();
        innerExp.free();
      }

      // Add variable declaration if needed
      if(tileDim == 1){
        if(varIsDeclared){
          expNode &newInitNode = *(iStatements[0]->getForStatement(0));

          expNode &ph = *(new expNode( *(newInitNode.sInfo) ));

          expNode::swap(newInitNode, ph);

          newInitNode.reserve(1);
          newInitNode.setLeaf(ph, 0);

          newInitNode.info = expType::declaration;
          newInitNode.getVariableInfoNode(0)->info |= (expType::declaration |
                                                       expType::type);
        }
      }
      else { // (1 < tileDim)
        statement &is = *(iStatements[0]);

        if(varIsDeclared)
          is.pushSourceLeftOf(is.statementStart,
                              (std::string) var + ";");

        statementNode *sn = is.statementStart;

        for(int dim = 0; dim < tileDim; ++dim){
          ss << var.name << "." << (char) ('x' + dim) << " = __occa_iTileVar" << dim << ';';

          is.pushSourceRightOf(sn,
                               ss.str());

          ss.str("");

          sn = sn->right;
        }
      }

      expNode::freeFlatHandle(csvCheckNode);
      expNode::freeFlatHandle(csvInitValueNode);
      expNode::freeFlatHandle(csvUpdateNode);
      expNode::freeFlatHandle(csvTileDims);

      delete [] checkIterOnLeft;
    }

    void parserBase::markKernelFunctions(statement &s){
      statementNode *snPos = s.statementStart;

      while(snPos){
        statement &s2 = *(snPos->value);

        if( !(s2.info & smntType::functionStatement) ||
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
      if((!(s.info & smntType::simpleStatement)    &&
          !(s.info & smntType::forStatement)       &&
          !(s.info & smntType::functionStatement)) ||
         // OCCA for's don't have arguments
         (s.info == smntType::occaFor))
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

          leaf.info  = expType::printValue;
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

        if(s.info & smntType::functionPrototype)
          prototypes[s.getFunctionName()] = true;

        statementPos = statementPos->right;
      }

      statementPos = globalScope->statementStart;

      while(statementPos){
        statement &s = *(statementPos->value);

        if(s.info & smntType::functionStatement){
          if(s.hasQualifier("occaKernel")){
            statementPos = statementPos->right;
            continue;
          }

          if(!s.hasQualifier("occaFunction"))
            s.addQualifier("occaFunction");

          if( !(s.info & smntType::functionDefinition) ){
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
      if((s.info != smntType::forStatement) ||
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

    void parserBase::checkOccaBarriers(statement &s){
      statementNode *statementPos = s.statementStart;

      while(statementPos){
        statement &s2 = *(statementPos->value);

        if(s2.info & smntType::ifStatement){
          if(s2.hasStatementWithBarrier()){
            OCCA_CHECK(false,
                       "Barriers are not allowed in conditional statements:\n" << s2);
          }
        }
        else
          checkOccaBarriers(s2);

        statementPos = statementPos->right;
      }
    }

    void parserBase::addOccaBarriers(){
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos){
        statement *s = statementPos->value;

        if(statementIsAKernel(*s)      && // Kernel
           (s->statementStart != NULL)){  // not empty

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

        if((occaType == (int) notAnOccaFor) ||
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

          if(snPos == lastLoop){
            if(warnForMissingBarriers){

              std::cout << "Warning: Placing a shared-memory barrier between:\n"
                        << "---[ A ]--------------------------------\n"
                        << *(firstLoop->value)
                        << "---[ B ]--------------------------------\n"
                        << *(lastLoop->value)
                        << "========================================\n";
            }

            s.pushSourceLeftOf(lastLoop, "occaBarrier(occaLocalMemFence);");
          }
        }

        statementPos = statementPos->right;
      }
    }

    bool parserBase::statementHasBarrier(statement &s){
      return s.hasBarrier();
    }

    void parserBase::updateConstToConstant(){
      statementNode *snPos = globalScope->statementStart;

      while(snPos){
        statement &s = *(snPos->value);

        if((s.info & smntType::declareStatement) &&
           (s.hasQualifier("occaConst"))){

          s.removeQualifier("occaConst");
          s.addQualifier("occaConstant");
        }

        snPos = snPos->right;
      }
    }

    void parserBase::addArgQualifiers(){
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos){
        statement &s = *(statementPos->value);

        if((s.info & smntType::functionDefinition) &&
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

          if((s.getFunctionArgCount() == 0) ||
             (s.getFunctionArgName(0) != "occaKernelInfoArg")){

            varInfo &arg0 = *(new varInfo());

            arg0.name = "occaKernelInfoArg";

            s.addFunctionArg(0, arg0);
          }
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::floatSharedAndExclusivesUp(statement &s){
      statementNode sn;

      // Get all shared and exclusive variables inside inner-loops
      appendSharedAndExclusives(s, &sn);

      statementNode *statementPos = sn.right;

      while(statementPos){
        statement &s2  = *(statementPos->value);
        statement *sUp = s2.up;

        // We're moving the definition else-where
        if(sUp){
          varInfo &var = s2.getDeclarationVarInfo(0);

          sUp->scopeVarMap.erase(var.name);

          sUp->removeStatement(s2);
        }

        // Find inner-most outer-for loop
        while(sUp){
          if((sUp->info == smntType::occaFor) &&
             statementIsOccaOuterFor(*sUp)){

            break;
          }

          sUp = sUp->up;
        }

        if(sUp){
          statementNode *sn3 = sUp->statementStart;

          // Skip exclusive and shared statements
          while(sn3){
            statement &s3 = *(sn3->value);

            if((!(s3.info & smntType::declareStatement)) ||
               (!s3.hasQualifier("exclusive") &&
                !s3.hasQualifier("occaShared"))){

              break;
            }

            sn3 = sn3->right;
          }

          const bool appendToEnd   = (sn3 == NULL);
          const bool appendToStart = (!appendToEnd) && (sn3->left == NULL);

          statementNode *sn2 = new statementNode(&s2);

          if(appendToStart){
            sn2->right                = sUp->statementStart;
            sUp->statementStart->left = sn2;

            sUp->statementStart = sn2;
          }
          else if(appendToEnd){
            sUp->statementEnd = lastNode(sUp->statementStart);

            sUp->statementEnd->right = sn2;
            sn2->left                = sUp->statementEnd;
          }
          else
            sn3->left->push(sn2);
        }

        statementPos = statementPos->right;
      }
    }

    statementNode* parserBase::appendSharedAndExclusives(statement &s,
                                                         statementNode *snTail,
                                                         bool isAppending){

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        statement &s2 = *(statementPos->value);

        if((s2.info & smntType::declareStatement)){
          if(isAppending &&
             (s2.hasQualifier("exclusive") ||
              s2.hasQualifier("occaShared"))){

            snTail = snTail->push(&s2);
          }
        }
        else{
          if(statementIsOccaInnerFor(s2))
            isAppending = true;

          snTail = appendSharedAndExclusives(s2, snTail, isAppending);
        }

        statementPos = statementPos->right;
      }

      return snTail;
    }

    void parserBase::modifyExclusiveVariables(statement &s){
      if( !(s.info & smntType::declareStatement)   ||
          (getStatementKernel(s) == NULL)    ||
          (statementKernelUsesNativeOCCA(s)) ||
          (!s.hasQualifier("exclusive")) ){

        return;
      }

      std::stringstream ss;

      const int argc = s.getDeclarationVarCount();

      //---[ Setup update statement ]---
      expNode newRoot = s.expRoot.clone();
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
      }
      //================================

      for(int i = 0; i < argc; ++i){
        varInfo &var = s.getDeclarationVarInfo(i);
        var.removeQualifier("exclusive");

        if(var.hasQualifier("occaConst"))
          var.removeQualifier("occaConst");

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
          OCCA_CHECK(var.stackPointerCount < 2,
                     "Only 1D exclusive arrays are supported:\n"
                     << "exclusive " << s);

          ss << var.stackExpRoots[0];
        }

        ss << ");";

        s.loadFromNode( splitAndLabelContent(ss.str()) );

        s.statementEnd->value->up = s.up;

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

    // [-] Missing
    void parserBase::modifyTextureVariables(){
      /*

        kernel void kern(texture float **tex){
          tex[j][i];
        }

        CPU:
        kernel void kern(int64 offsets[argc],
                         texture float tex,
                         sampler/textureInfo tex_info)
       */
    }

    statementNode* parserBase::splitKernelStatement(statementNode *snKernel,
                                                    kernelInfo &info){
      statement &sKernel = *(snKernel->value);

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
        bool lastOuter  = false;
        bool firstInner = true;

        //---[ Add kernel body ]----------
        if(((int) info.nestedKernels.size()) <= kID)
          break;

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
          else if(s2.isOccaOuterFor(forInfo)){
            if(lastOuter){
              lastOuter = false;
              break;
            }

            if(nest == 0)
              lastOuter = true;
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
        if(ksnOuter && ksnOuter->right){
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

        zeroOccaIdsFrom(ls);

        while(depIt != hostDepMap.end()){
          const int sID = (depIt->first);

          if(loopID < sID)
            break;

          statement &depS = *(sVec[sID]);

          zeroOccaIdsFrom(depS);

          if(blockStatement)
            blockStatement->addStatement(&depS);
          else
            newSNPos = newSNPos->push(&depS);

          ++depIt;
        }

        if(loopPos == loopOffsets[kID]){
          blockStatement = new statement(smntType::blockStatement,
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
      statementNode *ret = newSNEnd->right;

      sKernel.up->pushSourceLeftOf(snKernel , "#ifdef OCCA_LAUNCH_KERNEL");
      sKernel.up->pushSourceRightOf(snKernel, "#else");
      sKernel.up->pushSourceRightOf(newSNEnd, "#endif");

      return ret;
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

        if(s2.info == smntType::occaFor)
          tail = tail->push(new statementNode(&s2));

        if(getNestedLoops ||
           (s2.info != smntType::occaFor)){

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
        statement &ks2 = *(new statement(*this));

        // [occaOuterFor][#]
        // const int outerDim = atoi(ks.expRoot.value.c_str() + 12);

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

        // [--] Hard-code the OpenMP pragma without collapsing for now
        ss << "occaParallelFor0";
        // ss << "occaParallelFor" << outerDim;

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
      expNode nkNode = globalScope->createPlainExpNodeFrom("int *nestedKernels");
      arg.loadFrom(nkNode);

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
          varInfo &kernel = *(s.getFunctionVar());

          //---[ Setup Info ]-----------
          kernelInfo &info = *(new kernelInfo);

          info.name     = kernel.name;
          info.baseName = info.name;
          info.nestedKernels.push_back(&s);

          if(kernel.argumentCount){
            info.argumentInfos.reserve(kernel.argumentCount);

            for(int arg = 0; arg < kernel.argumentCount; ++arg){
              varInfo &varg = *(kernel.argumentVarInfos[arg]);

              argumentInfo argInfo;
              argInfo.isConst = varg.isConst();

              info.argumentInfos.push_back(argInfo);
            }
          }

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
        if(currentS->info == smntType::occaFor)
          break;

        currentS = currentS->up;
      }

      if(currentS == NULL)
        return "";

      return (std::string) s.expRoot;
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

      OCCA_CHECK(false,
                 "Error, outer-most loop doesn't contain obfuscate(\"" << tag << "\"):\n"
                 << s.expRoot);

      return -1;
    }

    void parserBase::splitDefineForVariable(varInfo &var){
      statement &origin = *(varUpdateMap[&var].value);

      // Ignore kernel arguments
      if(origin.info & smntType::functionStatement)
        return;

      int argc   = origin.getDeclarationVarCount();
      int argPos = 0;

      for(int i = 0; i < argc; ++i){
        varInfo &argVar = origin.getDeclarationVarInfo(i);

        if(&argVar == &var){
          argPos = i;
          break;
        }
      }

      if(argPos){
        statement &s        = origin.pushNewStatementLeft(smntType::declareStatement);
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

        statement &s        = origin.pushNewStatementRight(smntType::declareStatement);
        s.expRoot.info      = origin.expRoot.info;
        s.expRoot.leaves    = new expNode*[newLeafCount];
        s.expRoot.leafCount = newLeafCount;

        for(int i = 0; i < newLeafCount; ++i){
          varInfo &argVar = origin.getDeclarationVarInfo(argPos + 1 + i);

          s.expRoot.leaves[i]         = origin.expRoot.leaves[argPos + 1 + i];
          varUpdateMap[&argVar].value = &s;

          // Print out type for the new statement
          if(i == 0)
            s.expRoot.getVariableInfoNode(0)->info |= expType::type;
        }
      }

      origin.expRoot.leaves[0] = origin.expRoot.leaves[argPos];
      origin.expRoot.leafCount = 1;

      // Print out type for the new statement
      origin.expRoot.getVariableInfoNode(0)->info |= expType::type;
    }

    void parserBase::splitDefineAndInitForVariable(varInfo &var){
      statement &origin = *(varUpdateMap[&var].value);

      // Ignore kernel arguments
      if(origin.info & smntType::functionStatement)
        return;

      int argc = origin.getDeclarationVarCount();

      // Make sure var is the only variable
      if(1 < argc)
        splitDefineForVariable(var);

      // Return if [var] is not being initialized
      if(!origin.expRoot.variableHasInit(0))
        return;

      statement &s = origin.pushNewStatementRight(smntType::updateStatement);

      //---[ Swap Variables ]----------
      expNode &varNode = *(origin.expRoot.getVariableInfoNode(0));

      expNode &varNode2 = *(new expNode(s));
      varNode2.addVarInfoNode(0);
      varNode2.setVarInfo(var);

      expNode::swap(varNode, varNode2);
      //================================

      //---[ Swap ExpRoots ]------------
      s.expRoot.info = origin.expRoot.info;
      s.expRoot.addVarInfoNode(0);
      s.expRoot.setVarInfo(0, var);

      // Print out type for the new statement
      s.expRoot[0].info |= expType::type;

      // Swap and free old expNode
      expNode *tmp = &(origin.expRoot[0]);

      expNode::swap(origin.expRoot, origin.expRoot[0]);
      expNode::swap(origin.expRoot, s.expRoot);

      delete tmp;
      //================================

      if(s.expRoot.lastLeaf()->value != ";")
        s.expRoot.addNode(expType::endStatement, ";");

      s.addVariableToUpdateMap(var);
    }

    void parserBase::addInnerFors(statement &s){
      int innerDim = getKernelInnerDim(s);

      OCCA_CHECK(0 <= innerDim,
                 "OCCA Inner for-loop count could not be calculated");

      varInfoIdMap_t varInfoIdMap;
      int currentInnerID = 0;

      // Add inner for-loops
      addInnerForsTo(s, varInfoIdMap, currentInnerID, innerDim);
    }

    void parserBase::addInnerForsTo(statement &s,
                                    varInfoIdMap_t &varInfoIdMap,
                                    int &currentInnerID,
                                    const int innerDim){

      statementNode *ssStart = s.statementStart;
      statementNode *ssEnd   = lastNode(ssStart);

      statementNode *statementPos = ssStart;

      std::vector<statement*> sBreaks;
      std::vector<int> cInnerIDs;

      cInnerIDs.push_back(currentInnerID);

      while(statementPos){
        statement &s2 = *(statementPos->value);

        // Add inner-for inside the for/while loop
        if(s2.info & (smntType::forStatement |
                      smntType::whileStatement)){

          addInnerForsTo(s2, varInfoIdMap, currentInnerID, innerDim);
          sBreaks.push_back(&s2);
          cInnerIDs.push_back(++currentInnerID);
        }
        else if(s2.hasBarrier()){
          sBreaks.push_back(&s2);
          cInnerIDs.push_back(++currentInnerID);
        }

        statementPos = statementPos->right;
      }

      sBreaks.push_back(NULL);

      const int oldBreaks = sBreaks.size();
      int breaks = oldBreaks;

      // Start with first non-break statement
      for(int b = 0; b < breaks; ++b){
        if(ssStart &&
           (sBreaks[b] == ssStart->value)){

          ssStart = ssStart->right;

          --b;
          --breaks;
        }
        else
          break;
      }

      if(breaks == 0)
        return;

      // Remove first non-break statement info
      const int deltaBreaks = (oldBreaks - breaks);

      for(int b = 0; b < breaks; ++b){
        sBreaks[b]   = sBreaks[deltaBreaks + b];
        cInnerIDs[b] = cInnerIDs[deltaBreaks + b];
      }

      for(int b = 0; b < breaks; ++b){
        ssEnd = ssStart;

        if(ssEnd == NULL)
          break;

        const int cInnerID = cInnerIDs[b];

        // Find statements for inner-loop
        while(ssEnd &&
              ssEnd->value != sBreaks[b]){

          ssEnd = ssEnd->right;
        }

        statement *outerInnerS = NULL;
        statement *innerInnerS = NULL;

        // Add inner-for loops
        for(int i = 0; i <= innerDim; ++i){
          const int innerID = (innerDim - i);

          statement *newInnerS = new statement(smntType::occaFor,
                                               (outerInnerS ? outerInnerS : &s));

          if(outerInnerS == NULL){
            outerInnerS = newInnerS;
            innerInnerS = newInnerS;
          }

          newInnerS->expRoot.info   = expType::occaFor;
          newInnerS->expRoot.value  = "occaInnerFor";
          newInnerS->expRoot.value += ('0' + innerID);

          if(innerInnerS != newInnerS){
            innerInnerS->addStatement(newInnerS);
            innerInnerS = newInnerS;
          }
        }

        statementNode *sn = ssStart;

        while(ssStart != ssEnd){
          statement &s2 = *(ssStart->value);

          innerInnerS->addStatement(&s2);

          checkStatementForExclusives(s2,
                                      varInfoIdMap,
                                      cInnerID);

          // Save this SN for outerInnerS
          if(ssStart != sn)
            popAndGoRight(ssStart);
          else
            ssStart = ssStart->right;
        }

        // Move to the right of the break
        if(ssStart)
          ssStart = ssStart->right;

        sn->value = outerInnerS;

        // Skip breaks
        while(ssStart            &&
              ((b + 1) < breaks) &&
              ssStart->value == sBreaks[b + 1]){

          ssStart = ssStart->right;
          ++b;
        }
      } // for(breaks)
    }

    void parserBase::checkStatementForExclusives(statement &s,
                                                 varInfoIdMap_t &varInfoIdMap,
                                                 const int innerID){

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        checkStatementForExclusives(*(statementPos->value),
                                    varInfoIdMap,
                                    innerID);

        statementPos = statementPos->right;
      }

      // Find variables
      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        varInfo *sVar = NULL;

        // Check for variable
        if((flatRoot[i].info & expType::varInfo) &&
           (flatRoot[i].info & expType::declaration)){

          sVar = &(flatRoot[i].getVarInfo());
        }

        // Has variable
        if((sVar != NULL)                    &&
           !sVar->hasQualifier("occaShared") &&
           !sVar->hasQualifier("exclusive")){

          varInfoIdMapIterator it = varInfoIdMap.find(sVar);

          if(it != varInfoIdMap.end()){
            if(((it->second) != -1) &&
               ((it->second) != innerID)){

              splitDefineAndInitForVariable(*sVar);
              sVar->addQualifier("exclusive", 0);

              it->second = -1;
            }
          }
          else
            varInfoIdMap[sVar] = innerID;
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void parserBase::addOuterFors(statement &s){
      int outerDim = getKernelOuterDim(s);

      OCCA_CHECK(0 <= outerDim,
                 "OCCA Outer for-loop count could not be calculated");

      statement *sPos = &s;

      for(int o = outerDim; 0 <= o; --o){
        statement *newStatement = new statement(smntType::occaFor, &s);

        newStatement->expRoot.info   = expType::printValue;
        newStatement->expRoot.value  = "occaOuterFor";
        newStatement->expRoot.value += ('0' + o);

        newStatement->scopeVarMap = sPos->scopeVarMap;

        statementNode *sn = sPos->statementStart;

        while(sn){
          newStatement->addStatement(sn->value);

          sn->value->up = newStatement;

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

        if(sDown->info == smntType::blockStatement){
          sPos->scopeVarMap.insert(sDown->scopeVarMap.begin(),
                                   sDown->scopeVarMap.end());

          sPos->statementCount = 0;
          sPos->statementStart = sPos->statementEnd = NULL;

          statementNode *sn = sDown->statementStart;

          while(sn){
            sPos->addStatement(sn->value);

            sn->value->up = sPos;

            statementNode *sn2 = sn->right;
            delete sn;
            sn = sn2;
          }
        }
        else
          break;
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

    //---[ Operator Information ]-----------------
    varInfo* parserBase::hasOperator(const info_t expInfo,
                                     const std::string &op,
                                     varInfo &var){
      return NULL;
    }

    varInfo* parserBase::hasOperator(const info_t expInfo,
                                     const std::string &op,
                                     varInfo &varL,
                                     varInfo &varR){
      return NULL;
    }

    varInfo parserBase::thVarInfo(const info_t thType){
      varInfo var;

      const std::string str = typeHolder::typeToBaseTypeStr(thType);

      var.baseType = globalScope->hasTypeInScope(str);

      return var;
    }

    varInfo parserBase::thOperatorReturnType(const info_t expInfo,
                                             const std::string &op,
                                             const info_t thType){
      typeHolder th, ret;
      varInfo var;

      th.type = thType;

      if(expInfo & expType::L)
        ret = applyLOperator(op, th);
      else
        ret = applyROperator(th, op);

      var.baseType = globalScope->hasTypeInScope(ret.baseTypeStr());

      if(ret.isUnsigned())
        var.addQualifier("unsigned");

      if(ret.isALongLongInt())
        var.addQualifier("long");

      return var;
    }

    varInfo parserBase::thOperatorReturnType(const info_t expInfo,
                                             const std::string &op,
                                             const info_t thTypeL,
                                             const info_t thTypeR){
      varInfo var;

      if(op == "("){
        OCCA_CHECK(false,
                   "Cannot use () operator with void* (Example: 10(10))");
      }
      else if(op == "["){
        OCCA_CHECK(false,
                   "Cannot use [] operator with void* (Example: 10[10])");
      }
      else {
        typeHolder l, r;

        l.type = thTypeL;
        r.type = thTypeR;

        typeHolder ret = applyLROperator(l, op, r);

        var.baseType = globalScope->hasTypeInScope(ret.baseTypeStr());

        if(ret.isUnsigned())
          var.addQualifier("unsigned");

        if(ret.isALongLongInt())
          var.addQualifier("long");
      }

      return var;
    }
    //============================================

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

    expNode splitContent(const std::string &str, const int parsingLanguage){
      return splitContent(str.c_str(), parsingLanguage);
    }

    expNode splitContent(const char *cRoot, const int parsingLanguage){
      pushLanguage(parsingLanguage);

      const bool parsingFortran = (parsingLanguage & parserInfo::parsingFortran);

      const char *c = cRoot;

      int lineCount = 1 + countDelimiters(c, '\n');

      info_t status = readingCode;

      expNode allExp;
      allExp.addNodes(lineCount);
      int expPos = 0;

      while(*c != '\0'){
        const char *cEnd = readLine(c, parsingLanguage);

        std::string line = strip(c, cEnd - c, parsingLanguage);

        if(line.size()){
          if(parsingFortran &&
             (*c == 'c')){

            c = cEnd;
            continue;
          }

          if(status != insideCommentBlock){
            status = stripComments(line, parsingLanguage);

            strip(line, parsingLanguage);

            if(line.size()){
              allExp[expPos].value = line;
              ++expPos;
            }
          }
          else{
            status = stripComments(line, parsingLanguage);

            strip(line, parsingLanguage);

            if((status == finishedCommentBlock) && line.size()){
              allExp[expPos].value = line;
              ++expPos;
            }
          }
        }

        c = cEnd;
      }

      allExp.leafCount = expPos;

      popLanguage();

      return allExp;
    }

    expNode splitAndLabelContent(const std::string &str, const int parsingLanguage){
      return splitAndLabelContent(str.c_str(), parsingLanguage);
    }

    expNode splitAndLabelContent(const char *cRoot, const int parsingLanguage){
      expNode allExp = splitContent(cRoot, parsingLanguage);

      return labelCode(allExp);
    }

    expNode splitAndOrganizeContent(const std::string &str, const int parsingLanguage){
      return splitAndOrganizeContent(str.c_str(), parsingLanguage);
    }

    expNode splitAndOrganizeContent(const char *cRoot, const int parsingLanguage){
      expNode tmpExp, retExp;

      tmpExp = splitContent(cRoot, parsingLanguage);
      tmpExp = labelCode(tmpExp, parsingLanguage);

      retExp.loadFromNode(tmpExp);

      return retExp;
    }

    expNode& labelCode(expNode &allExp, const int parsingLanguage){
      pushLanguage(parsingLanguage);

      const bool parsingC       = (parsingLanguage & parserInfo::parsingC      );
      const bool parsingFortran = (parsingLanguage & parserInfo::parsingFortran);

      const bool usingLeaves = (allExp.leafCount != 0);
      const int lineCount    = (usingLeaves      ?
                                allExp.leafCount :
                                1);

      const bool addSpace = true; // For readability

      expNode node, *cNode = &node;

      for(int linePos = 0; linePos < lineCount; ++linePos){
        expNode &lineNode = (usingLeaves ? allExp[linePos] : allExp);

        const std::string &line = lineNode.value;
        const char *cLeft = line.c_str();

        while(*cLeft != '\0'){
          skipWhitespace(cLeft);

          const char *cRight = cLeft;

          bool loadString = isAString(cLeft);
          bool loadNumber = ((*cLeft != '-') &&
                             (*cLeft != '+') &&
                             isANumber(cLeft));

          if(loadString){ //-----------------------------------------------[ 1 ]
            skipString(cRight, parsingLanguage);

            cNode->addNode(expType::presetValue);
            cNode->lastNode().value = std::string(cLeft, (cRight - cLeft));

            cLeft = cRight;
          }
          else if(loadNumber){ //------------------------------------------[ 2 ]
            typeHolder th;
            th.load(cLeft);
            // skipNumber(cRight, parsingLanguage); [---]

            cNode->addNode(expType::presetValue);

            cNode->lastNode().value = (std::string) th;
            cNode->lastNode().info  = expType::firstPass | expType::presetValue;
          }
          else{ //---------------------------------------------------------[ 3 ]
            const int delimiterChars = isAWordDelimiter(cLeft, parsingLanguage);

            cNode->addNode();

            expNode &lastExpNode     = cNode->lastNode();
            std::string &lastNodeStr = lastExpNode.value;

            if(delimiterChars){ //---------------------------------------[ 3.1 ]
              if(parsingFortran){ //-------------------------------------[ 3.1.1 ]
                // Translate Fortran keywords
                std::string op(cLeft, delimiterChars);
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

                  lastNodeStr = upOp;
                }
                else if(upOp == "/="){
                  lastNodeStr = "!=";
                }
                else {
                  lastNodeStr = op;
                }
              }  //======================================================[ 3.1.1 ]
              else { //--------------------------------------------------[ 3.1.2 ]
                lastNodeStr = std::string(cLeft, delimiterChars);
              } //=======================================================[ 3.1.2 ]

              lastExpNode.info = (*keywordType)[lastExpNode.value];

              if(lastExpNode.info & expType::C){ //----------------------[ 3.1.3 ]
                if(charStartsSection(lastExpNode.value[0])){
                  cNode = &(cNode->lastNode());
                }
                else{
                  cNode->removeNode(-1);
                  cNode = cNode->up;
                }
              } //=======================================================[ 3.1.3 ]
              else if(lastExpNode.info & expType::macroKeyword){ //------[ 3.1.4 ]
                lastNodeStr = line;

                cLeft = line.c_str() + strlen(line.c_str()) - delimiterChars;
              } //=======================================================[ 3.1.4 ]

              cLeft += delimiterChars;
            } //=========================================================[ 3.1 ]
            else{ //-----------------------------------------------------[ 3.2 ]
              skipWord(cRight, parsingLanguage);

              std::string str(cLeft, (cRight - cLeft));
              keywordTypeMapIterator it;

              if(parsingFortran){
                std::string upStr = upString(str);

                it = keywordType->find(upStr);

                if(it != keywordType->end())
                  str = upStr;
              }
              else{
                it = keywordType->find(str);
              }

              lastNodeStr = str;

              if(it == keywordType->end())
                lastExpNode.info = expType::firstPass | expType::unknown;
              else{
                lastExpNode.info = it->second;

                if(parsingC){
                  if(checkLastTwoNodes(*cNode, "else", "if")){
                    mergeLastTwoNodes(*cNode);
                  }
                  else if((lastExpNode.info & expType::specialKeyword) &&
                          (lastExpNode.value == "__attribute__")){

                    skipWhitespace(cRight);
                    skipPair(cRight);

                    // [-] Early fix
                    if(cNode->leafCount)
                      cNode->removeNode(-1);
                    else if(cNode->up)
                      cNode = cNode->up;
                  }
                }
                else{
                  if(checkLastTwoNodes(*cNode, "else", "if"   , parsingLanguage) ||
                     checkLastTwoNodes(*cNode, "do"  , "while", parsingLanguage)){

                    mergeLastTwoNodes(*cNode, addSpace, parsingLanguage);
                  }
                  else if(checkLastTwoNodes(*cNode, "end" , "do"        , parsingLanguage) ||
                          checkLastTwoNodes(*cNode, "end" , "if"        , parsingLanguage) ||
                          checkLastTwoNodes(*cNode, "end" , "function"  , parsingLanguage) ||
                          checkLastTwoNodes(*cNode, "end" , "subroutine", parsingLanguage)){

                    mergeLastTwoNodes(*cNode, !addSpace, parsingLanguage);

                    expNode &mergedNode = (*cNode)[-1];

                    mergedNode.info = (*keywordType)[mergedNode.value];
                  }
                }
              }

              cLeft = cRight;
            } //=========================================================[ 3.2 ]
          } //===========================================================[ 3 ]
        }

        if(parsingFortran){
          cNode->addNode();
          cNode->lastNode().value = "\\n";
          cNode->lastNode().info  = expType::firstPass | expType::endStatement;
        }
      }

      expNode::swap(allExp, node);

      node.free();

      popLanguage();

      return allExp;
    }

    bool checkLastTwoNodes(expNode &node,
                           const std::string &leftValue,
                           const std::string &rightValue,
                           const int parsingLanguage){

      if(parsingLanguage & parserInfo::parsingC){
        return ((2 <= node.leafCount)   &&
                (node[-2].value == leftValue) &&
                (node[-1].value == rightValue));
      }

      return ((2 <= node.leafCount)              &&
              upStringCheck(node[-2].value, leftValue) &&
              upStringCheck(node[-1].value, rightValue));
    }

    void mergeLastTwoNodes(expNode &node,
                           const bool addSpace,
                           const int parsingLanguage){

      if(node.leafCount < 2)
        return;

      if(addSpace)
        node[-2].value += ' ';

      node[-2].value += node[-1].value;

      if(parsingLanguage & parserInfo::parsingFortran)
        node[-2].value = upString(node[-2].value);

      node.removeNode(-1);
    }

    expNode createExpNodeFrom(const std::string &source){
      return statement::createExpNodeFrom(source);
    }

    void loadKeywords(const int parsingLanguage){
      if(parsingLanguage & parserInfo::parsingC)
        loadCKeywords();
      else
        loadFortranKeywords();
    }

    void loadCKeywords(){
      initCKeywords();

      keywordType  = &cKeywordType;
      opPrecedence = &cOpPrecedence;

      for(int i = 0; i < maxOpLevels; ++i){
        opLevelMap[i] = &cOpLevelMap[i];
        opLevelL2R[i] = &cOpLevelL2R[i];
      }
    }

    void loadFortranKeywords(){
      initFortranKeywords();

      keywordType  = &fortranKeywordType;
      opPrecedence = &fortranOpPrecedence;

      for(int i = 0; i < maxOpLevels; ++i){
        opLevelMap[i] = &fortranOpLevelMap[i];
        opLevelL2R[i] = &fortranOpLevelL2R[i];
      }
    }

    void initCKeywords(){
      if(cKeywordsAreInitialized)
        return;

      cKeywordsAreInitialized = true;

      //---[ Operator Info ]--------------
      cKeywordType["!"]                  = expType::L;
      cKeywordType["%"]                  = expType::LR;
      cKeywordType["&"]                  = (expType::L | expType::LR | expType::qualifier);
      cKeywordType["("]                  = expType::C;
      cKeywordType[")"]                  = expType::C;
      cKeywordType["*"]                  = (expType::L | expType::LR | expType::qualifier);
      cKeywordType["+"]                  = (expType::L | expType::LR);
      cKeywordType[","]                  = expType::LR;
      cKeywordType["-"]                  = (expType::L | expType::LR);
      cKeywordType["."]                  = expType::LR;
      cKeywordType["/"]                  = expType::LR;
      cKeywordType[":"]                  = expType::endStatement;
      cKeywordType[";"]                  = expType::endStatement;
      cKeywordType["<"]                  = expType::LR;
      cKeywordType["="]                  = expType::LR;
      cKeywordType[">"]                  = expType::LR;
      cKeywordType["?"]                  = expType::LCR;
      cKeywordType["["]                  = expType::C;
      cKeywordType["]"]                  = expType::C;
      cKeywordType["^"]                  = (expType::LR | expType::qualifier);
      cKeywordType["{"]                  = expType::C;
      cKeywordType["|"]                  = expType::LR;
      cKeywordType["}"]                  = expType::C;
      cKeywordType["~"]                  = expType::L;
      cKeywordType["!="]                 = expType::LR;
      cKeywordType["%="]                 = expType::LR;
      cKeywordType["&&"]                 = expType::LR;
      cKeywordType["&="]                 = expType::LR;
      cKeywordType["*="]                 = expType::LR;
      cKeywordType["+="]                 = expType::LR;
      cKeywordType["++"]                 = expType::L_R;
      cKeywordType["-="]                 = expType::LR;
      cKeywordType["--"]                 = expType::L_R;
      cKeywordType["->"]                 = expType::LR;
      cKeywordType["/="]                 = expType::LR;
      cKeywordType["::"]                 = expType::LR;
      cKeywordType["<<"]                 = expType::LR;
      cKeywordType["<="]                 = expType::LR;
      cKeywordType["=="]                 = expType::LR;
      cKeywordType[">="]                 = expType::LR;
      cKeywordType[">>"]                 = expType::LR;
      cKeywordType["^="]                 = expType::LR;
      cKeywordType["|="]                 = expType::LR;
      cKeywordType["||"]                 = expType::LR;

      cKeywordType["#"]                  = expType::macroKeyword;

      cKeywordType["void"]               = expType::type;
      cKeywordType["__attribute__"]      = expType::type; // [--]

      cKeywordType["long"]               = (expType::qualifier | expType::type);
      cKeywordType["short"]              = (expType::qualifier | expType::type);
      cKeywordType["signed"]             = (expType::qualifier | expType::type);
      cKeywordType["unsigned"]           = (expType::qualifier | expType::type);

      cKeywordType["inline"]             = expType::qualifier;
      cKeywordType["static"]             = expType::qualifier;
      cKeywordType["extern"]             = expType::qualifier;

      cKeywordType["const"]              = (expType::qualifier | expType::occaKeyword);
      cKeywordType["restrict"]           = (expType::qualifier | expType::occaKeyword);
      cKeywordType["volatile"]           = (expType::qualifier | expType::occaKeyword);
      cKeywordType["aligned"]            = (expType::qualifier | expType::occaKeyword);
      cKeywordType["register"]           = expType::qualifier;

      cKeywordType["occaConst"]          = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaRestrict"]       = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaVolatile"]       = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaAligned"]        = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaConstant"]       = (expType::qualifier | expType::occaKeyword);

      cKeywordType["class"]              = (expType::struct_ | expType::qualifier);
      cKeywordType["enum"]               = (expType::struct_ | expType::qualifier);
      cKeywordType["union"]              = (expType::struct_ | expType::qualifier);
      cKeywordType["struct"]             = (expType::struct_ | expType::qualifier);
      cKeywordType["typedef"]            = expType::qualifier;

      //---[ Non-standard ]-------------
      cKeywordType["__attribute__"]      = (expType::qualifier | expType::specialKeyword);

      //---[ C++ ]----------------------
      cKeywordType["virtual"]            = expType::qualifier;

      cKeywordType["namespace"]          = (expType::type | expType::struct_);

      //---[ Constants ]------------------
      cKeywordType["..."]                = expType::printValue;
      cKeywordType["true"]               = expType::presetValue;
      cKeywordType["false"]              = expType::presetValue;

      //---[ Flow Control ]---------------
      cKeywordType["if"]                 = expType::flowControl;
      cKeywordType["else"]               = expType::flowControl;

      cKeywordType["for"]                = expType::flowControl;

      cKeywordType["do"]                 = expType::flowControl;
      cKeywordType["while"]              = expType::flowControl;

      cKeywordType["switch"]             = expType::flowControl;
      cKeywordType["case"]               = expType::specialKeyword;
      cKeywordType["default"]            = expType::specialKeyword;

      cKeywordType["break"]              = expType::specialKeyword;
      cKeywordType["continue"]           = expType::specialKeyword;
      cKeywordType["return"]             = expType::specialKeyword;
      cKeywordType["goto"]               = expType::specialKeyword;

      //---[ OCCA Keywords ]--------------
      cKeywordType["kernel"]             = (expType::qualifier | expType::occaKeyword);
      cKeywordType["texture"]            = (expType::qualifier | expType::occaKeyword);
      cKeywordType["shared"]             = (expType::qualifier | expType::occaKeyword);
      cKeywordType["exclusive"]          = (expType::qualifier | expType::occaKeyword);

      cKeywordType["occaKernel"]         = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaFunction"]       = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaDeviceFunction"] = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaPointer"]        = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaVariable"]       = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaShared"]         = (expType::qualifier | expType::occaKeyword);
      cKeywordType["occaFunctionShared"] = (expType::qualifier | expType::occaKeyword);

      cKeywordType["occaKernelInfoArg"]  = (expType::printValue | expType::occaKeyword);
      cKeywordType["occaKernelInfo"]     = (expType::printValue | expType::occaKeyword);

      cKeywordType["occaPrivate"]        = (expType::printValue | expType::occaKeyword);
      cKeywordType["occaPrivateArray"]   = (expType::printValue | expType::occaKeyword);

      cKeywordType["barrier"]            = (expType::printValue | expType::occaKeyword);
      cKeywordType["localMemFence"]      = (expType::printValue | expType::occaKeyword);
      cKeywordType["globalMemFence"]     = (expType::printValue | expType::occaKeyword);

      cKeywordType["occaBarrier"]        = (expType::printValue | expType::occaKeyword);
      cKeywordType["occaLocalMemFence"]  = (expType::printValue | expType::occaKeyword);
      cKeywordType["occaGlobalMemFence"] = (expType::printValue | expType::occaKeyword);

      cKeywordType["directLoad"]         = (expType::printValue | expType::occaKeyword);

      cKeywordType["atomicAdd"]          = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicSub"]          = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicSwap"]         = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicInc"]          = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicDec"]          = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicMin"]          = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicMax"]          = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicAnd"]          = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicOr"]           = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicXor"]          = (expType::printValue | expType::occaKeyword);

      cKeywordType["atomicAdd64"]        = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicSub64"]        = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicSwap64"]       = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicInc64"]        = (expType::printValue | expType::occaKeyword);
      cKeywordType["atomicDec64"]        = (expType::printValue | expType::occaKeyword);

      cKeywordType["occaInnerFor0"]      = expType::occaFor;
      cKeywordType["occaInnerFor1"]      = expType::occaFor;
      cKeywordType["occaInnerFor2"]      = expType::occaFor;

      cKeywordType["occaOuterFor0"]      = expType::occaFor;
      cKeywordType["occaOuterFor1"]      = expType::occaFor;
      cKeywordType["occaOuterFor2"]      = expType::occaFor;

      cKeywordType["occaInnerId0"]       = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaInnerId1"]       = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaInnerId2"]       = (expType::presetValue | expType::occaKeyword);

      cKeywordType["occaOuterId0"]       = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaOuterId1"]       = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaOuterId2"]       = (expType::presetValue | expType::occaKeyword);

      cKeywordType["occaGlobalId0"]      = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaGlobalId1"]      = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaGlobalId2"]      = (expType::presetValue | expType::occaKeyword);

      cKeywordType["occaInnerDim0"]      = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaInnerDim1"]      = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaInnerDim2"]      = (expType::presetValue | expType::occaKeyword);

      cKeywordType["occaOuterDim0"]      = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaOuterDim1"]      = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaOuterDim2"]      = (expType::presetValue | expType::occaKeyword);

      cKeywordType["occaGlobalDim0"]     = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaGlobalDim1"]     = (expType::presetValue | expType::occaKeyword);
      cKeywordType["occaGlobalDim2"]     = (expType::presetValue | expType::occaKeyword);

      cKeywordType["occaParallelFor0"]   = expType::specialKeyword;
      cKeywordType["occaParallelFor1"]   = expType::specialKeyword;
      cKeywordType["occaParallelFor2"]   = expType::specialKeyword;

      cKeywordType["occaUnroll"]         = expType::specialKeyword;

      //---[ CUDA Keywords ]--------------
      cKeywordType["threadIdx"]          = (expType::unknown | expType::cudaKeyword);
      cKeywordType["blockDim"]           = (expType::unknown | expType::cudaKeyword);
      cKeywordType["blockIdx"]           = (expType::unknown | expType::cudaKeyword);
      cKeywordType["gridDim"]            = (expType::unknown | expType::cudaKeyword);

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

        cKeywordType["occa"       + cmf] = expType::printValue;
        cKeywordType["occaFast"   + cmf] = expType::printValue;
        cKeywordType["occaNative" + cmf] = expType::printValue;
      }

      //---[ Operator Precedence ]--------
      cOpPrecedence[opHolder("::", expType::LR)]   = 0;

      // class(...), class{1,2,3}, static_cast<>(), func(), arr[]
      cOpPrecedence[opHolder("++", expType::R)]    = 1;
      cOpPrecedence[opHolder("--", expType::R)]    = 1;
      cOpPrecedence[opHolder("." , expType::LR)]   = 1;
      cOpPrecedence[opHolder("->", expType::LR)]   = 1;

      // (int) x, sizeof, new, new [], delete, delete []
      cOpPrecedence[opHolder("++", expType::L)]    = 2;
      cOpPrecedence[opHolder("--", expType::L)]    = 2;
      cOpPrecedence[opHolder("+" , expType::L)]    = 2;
      cOpPrecedence[opHolder("-" , expType::L)]    = 2;
      cOpPrecedence[opHolder("!" , expType::L)]    = 2;
      cOpPrecedence[opHolder("~" , expType::L)]    = 2;
      cOpPrecedence[opHolder("*" , expType::L)]    = 2;
      cOpPrecedence[opHolder("&" , expType::L)]    = 2;

      cOpPrecedence[opHolder(".*" , expType::LR)]  = 3;
      cOpPrecedence[opHolder("->*", expType::LR)]  = 3;

      cOpPrecedence[opHolder("*" , expType::LR)]   = 4;
      cOpPrecedence[opHolder("/" , expType::LR)]   = 4;
      cOpPrecedence[opHolder("%" , expType::LR)]   = 4;

      cOpPrecedence[opHolder("+" , expType::LR)]   = 5;
      cOpPrecedence[opHolder("-" , expType::LR)]   = 5;

      cOpPrecedence[opHolder("<<", expType::LR)]   = 6;
      cOpPrecedence[opHolder(">>", expType::LR)]   = 6;

      cOpPrecedence[opHolder("<" , expType::LR)]   = 7;
      cOpPrecedence[opHolder("<=", expType::LR)]   = 7;
      cOpPrecedence[opHolder(">=", expType::LR)]   = 7;
      cOpPrecedence[opHolder(">" , expType::LR)]   = 7;

      cOpPrecedence[opHolder("==", expType::LR)]   = 8;
      cOpPrecedence[opHolder("!=", expType::LR)]   = 8;

      cOpPrecedence[opHolder("&" , expType::LR)]   = 9;

      cOpPrecedence[opHolder("^" , expType::LR)]   = 10;

      cOpPrecedence[opHolder("|" , expType::LR)]   = 11;

      cOpPrecedence[opHolder("&&", expType::LR)]   = 12;

      cOpPrecedence[opHolder("||", expType::LR)]   = 13;

      cOpPrecedence[opHolder("?" , expType::LCR)]  = 14;
      cOpPrecedence[opHolder("=" , expType::LR)]   = 14;
      cOpPrecedence[opHolder("+=", expType::LR)]   = 14;
      cOpPrecedence[opHolder("-=", expType::LR)]   = 14;
      cOpPrecedence[opHolder("*=", expType::LR)]   = 14;
      cOpPrecedence[opHolder("/=", expType::LR)]   = 14;
      cOpPrecedence[opHolder("%=", expType::LR)]   = 14;
      cOpPrecedence[opHolder("<<=", expType::LR)]  = 14;
      cOpPrecedence[opHolder(">>=", expType::LR)]  = 14;
      cOpPrecedence[opHolder("&=", expType::LR)]   = 14;
      cOpPrecedence[opHolder("^=", expType::LR)]   = 14;
      cOpPrecedence[opHolder("|=", expType::LR)]   = 14;

      // 15: throw x

      cOpPrecedence[opHolder("," , expType::LR)]   = 16;

      cOpLevelMap[ 0]["::"]  = expType::LR;
      cOpLevelMap[ 1]["++"]  = expType::R;
      cOpLevelMap[ 1]["--"]  = expType::R;
      cOpLevelMap[ 1]["." ]  = expType::LR;
      cOpLevelMap[ 1]["->"]  = expType::LR;
      cOpLevelMap[ 2]["++"]  = expType::L;
      cOpLevelMap[ 2]["--"]  = expType::L;
      cOpLevelMap[ 2]["+" ]  = expType::L;
      cOpLevelMap[ 2]["-" ]  = expType::L;
      cOpLevelMap[ 2]["!" ]  = expType::L;
      cOpLevelMap[ 2]["~" ]  = expType::L;
      cOpLevelMap[ 2]["*" ]  = expType::L;
      cOpLevelMap[ 2]["&" ]  = expType::L;
      cOpLevelMap[ 3][".*" ] = expType::LR;
      cOpLevelMap[ 3]["->*"] = expType::LR;
      cOpLevelMap[ 4]["*" ]  = expType::LR;
      cOpLevelMap[ 4]["/" ]  = expType::LR;
      cOpLevelMap[ 4]["%" ]  = expType::LR;
      cOpLevelMap[ 5]["+" ]  = expType::LR;
      cOpLevelMap[ 5]["-" ]  = expType::LR;
      cOpLevelMap[ 6]["<<"]  = expType::LR;
      cOpLevelMap[ 6][">>"]  = expType::LR;
      cOpLevelMap[ 7]["<" ]  = expType::LR;
      cOpLevelMap[ 7]["<="]  = expType::LR;
      cOpLevelMap[ 7][">="]  = expType::LR;
      cOpLevelMap[ 7][">" ]  = expType::LR;
      cOpLevelMap[ 8]["=="]  = expType::LR;
      cOpLevelMap[ 8]["!="]  = expType::LR;
      cOpLevelMap[ 9]["&" ]  = expType::LR;
      cOpLevelMap[10]["^" ]  = expType::LR;
      cOpLevelMap[11]["|" ]  = expType::LR;
      cOpLevelMap[12]["&&"]  = expType::LR;
      cOpLevelMap[13]["||"]  = expType::LR;
      cOpLevelMap[14]["?" ]  = expType::LCR;
      cOpLevelMap[14]["=" ]  = expType::LR;
      cOpLevelMap[14]["+="]  = expType::LR;
      cOpLevelMap[14]["-="]  = expType::LR;
      cOpLevelMap[14]["*="]  = expType::LR;
      cOpLevelMap[14]["/="]  = expType::LR;
      cOpLevelMap[14]["%="]  = expType::LR;
      cOpLevelMap[14]["<<="] = expType::LR;
      cOpLevelMap[14][">>="] = expType::LR;
      cOpLevelMap[14]["&="]  = expType::LR;
      cOpLevelMap[14]["^="]  = expType::LR;
      cOpLevelMap[14]["|="]  = expType::LR;
      cOpLevelMap[16][","]   = expType::LR;

      /*---[ Future Ones ]----------------
        cKeywordType["using"]     = ;
        cKeywordType["namespace"] = ;
        cKeywordType["template"]  = ;
        ================================*/

      keywordTypeMapIterator it = cKeywordType.begin();

      while(it != cKeywordType.end()){
        it->second |= expType::firstPass;
        ++it;
      }
    }

    void initFortranKeywords(){
      if(fortranKeywordsAreInitialized)
        return;

      fortranKeywordsAreInitialized = true;

      //---[ Operator Info ]--------------
      fortranKeywordType["%"]  = expType::LR;
      fortranKeywordType["("]  = expType::C;
      fortranKeywordType[")"]  = expType::C;
      fortranKeywordType["(/"] = expType::C;
      fortranKeywordType["/)"] = expType::C;
      fortranKeywordType["**"] = (expType::LR);
      fortranKeywordType["*"]  = (expType::LR);
      fortranKeywordType["+"]  = (expType::L | expType::LR);
      fortranKeywordType[","]  = expType::LR;
      fortranKeywordType["-"]  = (expType::L | expType::LR);
      fortranKeywordType["/"]  = expType::LR;
      fortranKeywordType[";"]  = expType::endStatement;
      fortranKeywordType["<"]  = expType::LR;
      fortranKeywordType["="]  = expType::LR;
      fortranKeywordType[">"]  = expType::LR;
      fortranKeywordType["=>"] = expType::LR;
      fortranKeywordType["::"] = expType::LR;
      fortranKeywordType["<="] = expType::LR;
      fortranKeywordType["=="] = expType::LR;
      fortranKeywordType["/="] = expType::LR;
      fortranKeywordType[">="] = expType::LR;

      fortranKeywordType["#"]  = expType::macroKeyword;

      //---[ Types & Specifiers ]---------
      fortranKeywordType["int"]    = expType::type;
      fortranKeywordType["bool"]   = expType::type;
      fortranKeywordType["char"]   = expType::type;
      fortranKeywordType["long"]   = expType::type;
      fortranKeywordType["short"]  = expType::type;
      fortranKeywordType["float"]  = expType::type;
      fortranKeywordType["double"] = expType::type;

      fortranKeywordType["void"]   = expType::type;

      fortranKeywordType["true"]  = expType::presetValue;
      fortranKeywordType["false"] = expType::presetValue;

      //---[ Types and Specifiers ]-----
      fortranKeywordType["INTEGER"]   = expType::type;
      fortranKeywordType["LOGICAL"]   = expType::type;
      fortranKeywordType["REAL"]      = expType::type;
      fortranKeywordType["PRECISION"] = expType::type;
      fortranKeywordType["COMPLEX"]   = expType::type;
      fortranKeywordType["CHARACTER"] = expType::type;

      std::string suffix[5] = {"2", "3", "4", "8", "16"};

      for(int i = 0; i < 5; ++i){
        fortranKeywordType[std::string("INTEGER") + suffix[i]] = expType::type;
        fortranKeywordType[std::string("REAL")    + suffix[i]] = expType::type;
      }

      fortranKeywordType["FUNCTION"]      = expType::specialKeyword;
      fortranKeywordType["SUBROUTINE"]    = expType::specialKeyword;
      fortranKeywordType["CALL"]          = expType::specialKeyword;

      fortranKeywordType["DOUBLE"]        = expType::qualifier;

      fortranKeywordType["ALLOCATABLE"]   = expType::qualifier;
      fortranKeywordType["AUTOMATIC"]     = expType::qualifier;
      fortranKeywordType["DIMENSION"]     = expType::qualifier;
      fortranKeywordType["EXTERNAL"]      = expType::qualifier;
      fortranKeywordType["IMPLICIT"]      = expType::qualifier;
      fortranKeywordType["INTENT"]        = expType::qualifier;
      fortranKeywordType["INTRINSIC"]     = expType::qualifier;
      fortranKeywordType["OPTIONAL"]      = expType::qualifier;
      fortranKeywordType["PARAMETER"]     = expType::qualifier;
      fortranKeywordType["POINTER"]       = expType::qualifier;
      fortranKeywordType["PRIVATE"]       = expType::qualifier;
      fortranKeywordType["PUBLIC"]        = expType::qualifier;
      fortranKeywordType["RECURSIVE"]     = expType::qualifier;
      fortranKeywordType["SAVE"]          = expType::qualifier;
      fortranKeywordType["STATIC"]        = expType::qualifier;
      fortranKeywordType["TARGET"]        = expType::qualifier;
      fortranKeywordType["VOLATILE"]      = expType::qualifier;

      fortranKeywordType["NONE"]          = expType::specialKeyword;

      fortranKeywordType["KERNEL"]        = expType::qualifier;
      fortranKeywordType["DEVICE"]        = expType::qualifier;
      fortranKeywordType["SHARED"]        = expType::qualifier;
      fortranKeywordType["EXCLUSIVE"]     = expType::qualifier;

      fortranKeywordType["DIRECTLOAD"]    = (expType::printValue | expType::occaKeyword);

      //---[ Atomics ]--------------------
      fortranKeywordType["ATOMICADD"]     = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICSUB"]     = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICSWAP"]    = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICINC"]     = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICDEC"]     = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICMIN"]     = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICMAX"]     = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICAND"]     = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICOR"]      = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICXOR"]     = (expType::printValue | expType::occaKeyword);

      fortranKeywordType["ATOMICADD64"]   = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICSUB64"]   = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICSWAP64"]  = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICINC64"]   = (expType::printValue | expType::occaKeyword);
      fortranKeywordType["ATOMICDEC64"]   = (expType::printValue | expType::occaKeyword);

      //---[ Constants ]------------------
      fortranKeywordType[":"]             = expType::printValue;

      //---[ Flow Control ]---------------
      fortranKeywordType["DO"]            = expType::flowControl;
      fortranKeywordType["WHILE"]         = expType::flowControl;
      fortranKeywordType["DO WHILE"]      = expType::flowControl;

      fortranKeywordType["IF"]            = expType::flowControl;
      fortranKeywordType["THEN"]          = expType::flowControl;
      fortranKeywordType["ELSE IF"]       = expType::flowControl;
      fortranKeywordType["ELSE"]          = expType::flowControl;

      fortranKeywordType["ENDDO"]         = expType::endStatement;
      fortranKeywordType["ENDIF"]         = expType::endStatement;
      fortranKeywordType["ENDFUNCTION"]   = expType::endStatement;
      fortranKeywordType["ENDSUBROUTINE"] = expType::endStatement;

      fortranKeywordType["RETURN"]        = expType::specialKeyword;

      std::string mathFunctions[16] = {
        "SQRT", "SIN"  , "ASIN" ,
        "SINH", "ASINH", "COS"  ,
        "ACOS", "COSH" , "ACOSH",
        "TAN" , "ATAN" , "TANH" ,
        "ATANH", "EXP" , "LOG2" ,
        "LOG10"
      };

      for(int i = 0; i < 16; ++i)
        fortranKeywordType[ mathFunctions[i] ] = expType::printValue;

      //---[ Operator Precedence ]--------
      fortranOpPrecedence[opHolder("%" , expType::LR)]  = 0;
      fortranOpPrecedence[opHolder("=>", expType::LR)]  = 0;

      fortranOpPrecedence[opHolder("**", expType::LR)]  = 1;
      fortranOpPrecedence[opHolder("//", expType::LR)]  = 1;

      fortranOpPrecedence[opHolder("+", expType::L)]    = 2;
      fortranOpPrecedence[opHolder("-", expType::L)]    = 2;

      fortranOpPrecedence[opHolder("*", expType::LR)]   = 3;
      fortranOpPrecedence[opHolder("/", expType::LR)]   = 3;

      fortranOpPrecedence[opHolder("+", expType::LR)]   = 4;
      fortranOpPrecedence[opHolder("-", expType::LR)]   = 4;

      fortranOpPrecedence[opHolder("<" , expType::LR)]  = 5;
      fortranOpPrecedence[opHolder("<=", expType::LR)]  = 5;
      fortranOpPrecedence[opHolder(">=", expType::LR)]  = 5;
      fortranOpPrecedence[opHolder(">" , expType::LR)]  = 5;

      fortranOpPrecedence[opHolder("!", expType::LR)]   = 6;
      fortranOpPrecedence[opHolder("&&", expType::LR)]  = 7;
      fortranOpPrecedence[opHolder("||", expType::LR)]  = 8;

      fortranOpPrecedence[opHolder("==" , expType::LR)] = 9;
      fortranOpPrecedence[opHolder("!=", expType::LR)]  = 9;

      fortranOpPrecedence[opHolder("=" , expType::LR)]  = 10;

      fortranOpPrecedence[opHolder("," , expType::LR)]  = 11;

      fortranOpLevelMap[0]["%"]   = expType::LR;
      fortranOpLevelMap[0]["=>"]  = expType::LR;
      fortranOpLevelMap[1]["**"]  = expType::LR;
      fortranOpLevelMap[1]["//"]  = expType::LR;
      fortranOpLevelMap[2]["+"]   = expType::L;
      fortranOpLevelMap[2]["-"]   = expType::L;
      fortranOpLevelMap[3]["*"]   = expType::LR;
      fortranOpLevelMap[3]["/"]   = expType::LR;
      fortranOpLevelMap[4]["+"]   = expType::LR;
      fortranOpLevelMap[4]["-"]   = expType::LR;
      fortranOpLevelMap[5]["<"]   = expType::LR;
      fortranOpLevelMap[5]["<="]  = expType::LR;
      fortranOpLevelMap[5][">="]  = expType::LR;
      fortranOpLevelMap[5][">"]   = expType::LR;
      fortranOpLevelMap[6]["!"]   = expType::LR;
      fortranOpLevelMap[7]["&&"]  = expType::LR;
      fortranOpLevelMap[8]["||"]  = expType::LR;
      fortranOpLevelMap[9]["=="]  = expType::LR;
      fortranOpLevelMap[9]["!="]  = expType::LR;
      fortranOpLevelMap[10]["="]  = expType::LR;
      fortranOpLevelMap[11][","]  = expType::LR;

      keywordTypeMapIterator it = fortranKeywordType.begin();

      while(it != fortranKeywordType.end()){
        it->second |= expType::firstPass;
        ++it;
      }
    }

    //---[ OCCA Loop Info ]-------------
    occaLoopInfo::occaLoopInfo(statement &s,
                               const int parsingLanguage_,
                               const std::string &tag){
      parsingLanguage = parsingLanguage_;

      lookForLoopFrom(s, tag);
    }

    void occaLoopInfo::lookForLoopFrom(statement &s,
                                       const std::string &tag){
      sInfo = &s;

      while(sInfo){
        if((sInfo->info & smntType::forStatement) &&
           (sInfo->getForStatementCount() > 3)){

          OCCA_CHECK(sInfo->getForStatementCount() <= 4,
                     "More than 4 statements for:\n  " << sInfo->expRoot);

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

      sInfo->info = smntType::occaFor;

      expNode &node1   = *(sInfo->getForStatement(0));
      expNode &node2   = *(sInfo->getForStatement(1));
      expNode &node3   = *(sInfo->getForStatement(2));
      std::string arg4 = (std::string) *(sInfo->getForStatement(3));

      // Fortran-loading is easier
      if(parsingLanguage & parserInfo::parsingFortran){
        //---[ Node 4 Check ]---
        // If it has a fourth argument, make sure it's the correct one
        OCCA_CHECK(isAnOccaTag(arg4),
                   "Wrong 4th statement for:\n  " << sInfo->expRoot);

        return;
      }

      //---[ Node 1 Check ]---
      OCCA_CHECK((node1.info == expType::declaration) &&
                 (node1.getVariableCount() == 1)      &&
                 node1.variableHasInit(0),

                 "Wrong 1st statement for:\n  " << sInfo->expRoot);

      varInfo &iterVar = node1.getVariableInfoNode(0)->getVarInfo();

      std::string &iter = iterVar.name;

      if( !iterVar.hasQualifier("occaConst") )
        iterVar.addQualifier("occaConst");

      //---[ Node 2 Check ]---
      OCCA_CHECK((node2.leafCount == 1) &&
                 ((node2[0].value == "<=") ||
                  (node2[0].value == "<" ) ||
                  (node2[0].value == ">" ) ||
                  (node2[0].value == ">=")),

                 "Wrong 2nd statement for:\n  " << sInfo->expRoot);

      if(parsingLanguage & parserInfo::parsingC){
        OCCA_CHECK((node2[0][0].value == iter) ||
                   (node2[0][1].value == iter),

                   "Wrong 2nd statement for:\n  " << sInfo->expRoot);
      }

      //---[ Node 3 Check ]---
      OCCA_CHECK((node3.leafCount == 1) &&
                 ((node3[0].value == "++") ||
                  (node3[0].value == "--") ||
                  (node3[0].value == "+=") ||
                  (node3[0].value == "-=")),

                 "Wrong 3rd statement for:\n  " << sInfo->expRoot);

      OCCA_CHECK((node3[0][0].value == iter) ||
                 (node3[0][1].value == iter),

                 "Wrong 3rd statement for:\n  " << sInfo->expRoot);

      //---[ Node 4 Check ]---
      // If it has a fourth argument, make sure it's the correct one
      OCCA_CHECK(isAnOccaTag(arg4),
                 "Wrong 4th statement for:\n  " << sInfo->expRoot);
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

      if(parsingLanguage & parserInfo::parsingC){
        varInfo &iterVar = node1.getVariableInfoNode(0)->getVarInfo();

        iter  = iterVar.name;
        start = *(node1.getVariableInitNode(0));
      }
      else{
        iter  = node1[0][0].getVarInfo().name;
        start = node1[0][1].getVarInfo().name;
      }
    }

    void occaLoopInfo::getLoopNode2Info(std::string &bound,
                                        std::string &iterCheck){

      expNode &node2 = *(sInfo->getForStatement(1));

      iterCheck = node2[0].value;

      if(parsingLanguage & parserInfo::parsingC){
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
        if(node3[0][0].getVarInfo().name == iter)
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
      node1.printOn(ss, "", (expFlag::noSemicolon |
                             expFlag::noNewline));

      return ss.str();
    }
    //==================================
  };
};
