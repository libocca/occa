/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include "occa/parser/parser.hpp"
#include "occa/kernel.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  namespace parserNS {
    intVector_t loadedLanguageVec;

    int loadedLanguage() {
      return loadedLanguageVec.back();
    }

    void pushLanguage(const int language) {
      loadedLanguageVec.push_back(language);
      loadKeywords(language);
    }

    int popLanguage() {
      const int ret = loadedLanguageVec.back();

      loadedLanguageVec.pop_back();

      if (loadedLanguageVec.size())
        loadKeywords(loadedLanguageVec.back());

      return ret;
    }

    parserBase::parserBase() {
      env::initialize();

      parsingLanguage = parserInfo::parsingC;

      macrosAreInitialized = false;

      globalScope       = new statement(*this);
      globalScope->info = smntType::namespaceStatement;
    }

    parserBase::~parserBase() {
      if (globalScope) {
        delete globalScope;
        globalScope = NULL;
      }
    }

    const std::string parserBase::parseFile(const std::string &filename_,
                                            const occa::properties &properties_) {

      filename = filename_;

      setProperties(properties_);

      //---[ Language ]-------
      parsingLanguage = parserInfo::parsingC;
      pushLanguage(parsingLanguage);

      //---[ Mode ]-----------
      OCCA_ERROR("Compilation mode must be passed to the parser",
                 properties.has("mode"));

      std::string content = assembleHeader(properties);
      content += io::read(filename);
      content += (std::string) properties["footer"];

      return parseSource(content.c_str());
    }

    const std::string parserBase::parseSource(const char *cRoot) {
      expNode allExp = splitAndPreprocessContent(cRoot, parsingLanguage);
      // allExp.print();
      // throw 1;

      loadLanguageTypes();

      globalScope->loadAllFromNode(allExp, parsingLanguage);
      // std::cout << (std::string) *globalScope;
      // throw 1;

      reorderLoops();
      retagOccaLoops();

      applyToAllStatements(*globalScope, &parserBase::splitTileOccaFors);

      markKernelFunctions();
      labelNativeKernels();

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
      // std::cout << (std::string) *globalScope;
      // throw 1;

      loadKernelInfos();

      applyToAllStatements(*globalScope, &parserBase::modifyExclusiveVariables);

      return (std::string) *globalScope;
    }

    //---[ Parser Warnings ]------------
    void parserBase::setProperties(const occa::properties &properties_) {
      properties = properties_;

      const std::string &mode = properties["mode"];
      _compilingForCPU = ((mode == "Serial")   ||
                          (mode == "Pthreads") ||
                          (mode == "OpenMP"));

      _warnForConditionalBarriers  = properties.get("warn-for-conditional-barriers", false);
      _insertBarriersAutomatically = properties.get("automate-add-barriers"        , true);
    }

    bool parserBase::compilingForCPU() {
      return false;
      // return _compilingForCPU;
    }

    bool parserBase::warnForConditionalBarriers() {
      return _warnForConditionalBarriers;
    }

    bool parserBase::insertBarriersAutomatically() {
      return _insertBarriersAutomatically;
    }
    //==================================

    //---[ Macro Parser Functions ]-------
    std::string parserBase::getMacroName(const char *&c) {
      const char *cStart = c;
      skipWord(cStart);
      skipWhitespace(cStart);
      c = cStart;

      while((*c != '\0') &&
            (*c != '(')  &&
            !isWhitespace(*c)) {

        if ((c[0] == '#') && (c[1] == '<')) {
          while((c[0] != '\0') &&
                ((c[0] != '#') || (c[1] != '>'))) {

            ++c;
          }

          c += 2;
        }
        else
          ++c;
      }

      std::string name(cStart, c - cStart);

      if (macroMap.find(name) == macroMap.end())
        applyMacros(name);

      return name;
    }

    std::string parserBase::getMacroIncludeFile(const char *&c) {
      const char *cStart = c;
      skipWord(cStart);

      c = cStart;

      while((*c != '\n') &&
            (*c != '\0')) {
        ++c;
      }

      const std::string iFilename = compressAllWhitespace(cStart, c - cStart);
      io::fileOpener &opener = io::fileOpener::get(iFilename);
      return opener.expand(iFilename);
    }

    typeHolder parserBase::evaluateMacroStatement(const char *c) {
      return evaluateString(c, this);
    }

    bool parserBase::evaluateMacroBoolStatement(const char *c) {
      typeHolder th = evaluateMacroStatement(c);

      return th.to<bool>();
    }

    void parserBase::loadMacroInfo(macroInfo &info, const char *&c) {
      const bool hasWhitespace = isWhitespace(*c);

      skipWhitespace(c);
      info.reset();

      if (*c == '\0') {
        return;
      }

      if ((*c != '(') || hasWhitespace) {
        const size_t chars = strlen(c);

        info.parts[0] = compressAllWhitespace(c, chars);
        c += chars;

        return;
      }

      int partPos = 0;
      info.isAFunction = true;

      ++c; // '('

      typedef std::map<std::string,int> macroArgMap_t;
      typedef macroArgMap_t::iterator macroArgMapIterator;
      macroArgMap_t macroArgMap;

      while(*c != '\0') {
        const char *cStart = c;
        skipTo(c, ",)");

        const std::string macroArgName = compressAllWhitespace(cStart, c - cStart);

        if (macroArgName.size()) {
          OCCA_ERROR("Macro [" << info.name << "] has arguments after variadic ... argument",
                     !info.hasVarArgs);
          if (macroArgName != "...") {
            macroArgMap[macroArgName] = (info.argc++);
          }
          else {
            info.hasVarArgs = true;
            macroArgMap["__VA_ARGS__"] = macroInfo::VA_ARGS_POS;
          }
        }
        else {
          OCCA_ERROR("Macro [" << info.name << "] has an argument without a name",
                     (*c == ')') && (info.argc == 0));
        }

        if (*c == ')')
          break;

        ++c;
      }

      ++c; // ')'

      if (isWhitespace(*c)) {
        info.parts[partPos] += ' ';
        skipWhitespace(c);
      }

      while(*c != '\0') {
        const char *cStart = c;

        if (isAString(c)) {
          skipString(c, parsingLanguage);

          info.parts[partPos] += std::string(cStart, (c - cStart));
          continue;
        }

        const int delimiterChars = skipWord(c);

        std::string word = std::string(cStart, c - cStart);

        macroArgMapIterator it = macroArgMap.find(word);

        if (it == macroArgMap.end()) {
          info.parts[partPos] += word;
        }
        else {
          info.argBetweenParts.push_back(it->second);
          info.parts.push_back("");
          ++partPos;
        }

        cStart = c;
        c += delimiterChars;

        if (cStart != c)
          info.parts[partPos] += std::string(cStart, c - cStart);

        if (isWhitespace(*c)) {
          info.parts[partPos] += ' ';
          skipWhitespace(c);
        }
      }
    }

    int parserBase::loadMacro(expNode &allExp, int leafPos, const int state) {
      return loadMacro(allExp, leafPos, allExp[leafPos].value, state);
    }

    int parserBase::loadMacro(const std::string &line, const int state) {
      expNode dummyExpRoot;

      return loadMacro(dummyExpRoot, -1, line, state);
    }

    int parserBase::loadMacro(expNode &allExp, int leafPos,
                              const std::string &line, const int state) {

      const char *c = (line.c_str() + 1); // line[0] = #

      while(*c != '\0') {
        skipWhitespace(c);
        const char *cEnd = c;
        skipToWhitespace(cEnd);

        if (stringsAreEqual(c, (cEnd - c), "if")) {
          if (state & ignoring) {
            return (startHash | ignoreUntilEnd);
          }
          c = cEnd;

          bool isTrue = evaluateMacroBoolStatement(c);

          if (isTrue) {
            return (startHash | readUntilNextHash);
          } else {
            return (startHash | ignoreUntilNextHash);
          }
        }

        else if (stringsAreEqual(c, (cEnd - c), "elif")) {
          if ((state & readUntilNextHash) || (state & ignoreUntilEnd))
            return (ignoreUntilEnd);

          c = cEnd;

          bool isTrue = evaluateMacroBoolStatement(c);

          if (isTrue)
            return (readUntilNextHash);
          else
            return (ignoreUntilNextHash);
        }

        else if (stringsAreEqual(c, (cEnd - c), "else")) {
          if ((state & readUntilNextHash) || (state & ignoreUntilEnd)) {
            return (ignoreUntilEnd);
          } else {
            return (readUntilNextHash);
          }
        }

        else if (stringsAreEqual(c, (cEnd - c), "ifdef")) {
          std::string name = getMacroName(c);

          if (macroMap.find(name) != macroMap.end())
            return (startHash | readUntilNextHash);
          else
            return (startHash | ignoreUntilNextHash);
        }

        else if (stringsAreEqual(c, (cEnd - c), "ifndef")) {
          std::string name = getMacroName(c);

          if (macroMap.find(name) != macroMap.end())
            return (startHash | ignoreUntilNextHash);
          else
            return (startHash | readUntilNextHash);
        }

        else if (stringsAreEqual(c, (cEnd - c), "endif")) {
          return (doneIgnoring);
        }

        else if (stringsAreEqual(c, (cEnd - c), "define")) {
          if (state & ignoring)
            return state;

          std::string name = getMacroName(c);
          int pos;

          if (macroMap.find(name) == macroMap.end()) {
            macroInfo tmpMacro;

            pos = macros.size();
            macros.push_back(tmpMacro);
            macroMap[name] = pos;
          } else {
            pos = macroMap[name];
          }

          macroInfo &info = macros[pos];
          info.name = name;

          loadMacroInfo(info, c);

          return (state);
        }

        else if (stringsAreEqual(c, (cEnd - c), "undef")) {
          if (state & ignoring)
            return state;

          std::string name = getMacroName(c);

          if (macroMap.find(name) != macroMap.end())
            macroMap.erase(name);

          return state;
        }

        else if (stringsAreEqual(c, (cEnd - c), "include")) {
          if (state & ignoring)
            return state;

          // Nothing to edit, just keep the #include for the compiler
          if (leafPos == -1)
            return (state | keepMacro);

          std::string includeFile = getMacroIncludeFile(c);
          includeFile = io::filename(includeFile);

          if (includeFile == "")
            return (state);

          const char *cRoot = io::c_read(includeFile);

          expNode includeExpRoot = splitContent(cRoot, parsingLanguage);

          delete [] cRoot;

          // Empty include file
          if (includeExpRoot.leafCount == 0)
            return (state);

          leafPos = allExp.insertExpAt(includeExpRoot, leafPos + 1);

          includeExpRoot.free();

          return (state);
        }

        else if (stringsAreEqual(c, (cEnd - c), "pragma"))
          return (state | keepMacro);

        else
          return (state | keepMacro);

        c = cEnd;
      }

      return state;
    }

    void parserBase::applyMacros(std::string &line) {
      const char *c = line.c_str();
      std::string newLine = "";

      bool foundMacro = false;

      while(*c != '\0') {
        const char *cStart = c;

        if (isAString(c)) {
          skipString(c, parsingLanguage);

          newLine += std::string(cStart, (c - cStart));
          continue;
        }

        int delimiterChars = skipWord(c);

        std::string word = std::string(cStart, c - cStart);

        macroMapIterator it = macroMap.find(word);

        while(delimiterChars == 2) {
          //---[ #< #> ]----------------
          if (stringsAreEqual(c, delimiterChars, "#<")) {
            c += 2;
            cStart = c;

            while((c[0] != '\0') &&
                  ((c[0] != '#') || (c[1] != '>'))) {

              ++c;
            }

            std::string expr(cStart, c - cStart);
            const char *c_expr = expr.c_str();

            std::string expr2 = (std::string) evaluateMacroStatement(c_expr);

            while(expr != expr2) {
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
          else if (stringsAreEqual(c, delimiterChars, "##")) {
            c += 2;

            cStart = c;
            delimiterChars = skipWord(c);

            std::string word2 = std::string(cStart, c - cStart);

            it = macroMap.find(word2);

            if (it != macroMap.end()) {
              macroInfo &info = macros[it->second];
              word += info.parts[0];
            }
            else {
              word += word2;
            }
          }
          else
            break;
          //============================
        }

        it = macroMap.find(word);

        bool wordIsAMacro = false;
        macroInfo *info_ = (it != macroMap.end()) ? &(macros[it->second]) : NULL;

        if (info_) {
          macroInfo &info = *info_;

          cStart = c;
          skipWhitespace(cStart);
          const bool foundAFunction = (*cStart == '(');

          wordIsAMacro = (!info.isAFunction || foundAFunction);
        }

        if (wordIsAMacro) {
          macroInfo &info = *info_;
          foundMacro = true;

          if (!info.isAFunction) {
            newLine += info.parts[0];
          }
          else {
            std::vector<std::string> args;

            cStart = c + 1;
            skipWhitespace(cStart);

            skipPair(c);

            const char *cEnd = c;
            c = cStart;

            OCCA_ERROR("Missing ')' in ["
                       << info.name
                       << "(" << std::string(cStart, cEnd - cStart - 1) << ")]",
                       *c != '\0');

            while(c < cEnd) {
              if (*c == ',') {
                args.push_back( compressAllWhitespace(cStart, c - cStart) );
                cStart = ++c; // Skip the [,]
              }
              else {
                if (segmentPair(*c))
                  skipPair(c);
                else if (isAString(c))
                  skipString(c);
                else
                  ++c;
              }
            }

            c = cEnd;

            if (cStart < (cEnd - 1)) {
              args.push_back( std::string(cStart, cEnd - cStart - 1) );
            }

            const int argCount = (int) args.size();
            for (int i = 0; i < argCount; ++i) {
              applyMacros(args[i]);
            }
            std::string funcResult = info.applyArgs(args);
            applyMacros(funcResult);
            newLine += funcResult;
          }
        }
        else {
          newLine += word;
        }

        cStart = c;

        if (*c != '\0')
          c += delimiterChars;

        if (cStart != c)
          newLine += std::string(cStart, c - cStart);

        if (isWhitespace(*c)) {
          newLine += ' ';
          skipWhitespace(c);
        }
      }

      line = newLine;

      if (foundMacro)
        applyMacros(line);
    }

    void parserBase::preprocessMacros(expNode &allExp) {
      std::stack<int> statusStack;
      std::vector<int> linesIgnored;

      int currentState = doNothing;

      for (int linePos = 0; linePos < (allExp.leafCount); ++linePos) {
        std::string &line = allExp[linePos].value;
        bool ignoreLine = false;

        if (line[0] == '#') {
          const int oldState = currentState;

          currentState = loadMacro(allExp, linePos, currentState);

          if (currentState & keepMacro)
            currentState &= ~keepMacro;
          else
            ignoreLine = true;

          // Nested #if's
          if (currentState & startHash) {
            currentState &= ~startHash;
            statusStack.push(oldState);
          }

          if (currentState & doneIgnoring) {
            if (statusStack.size()) {
              currentState = statusStack.top();
              statusStack.pop();
            }
            else
              currentState = doNothing;
          }
        }
        else {
          if (!(currentState & ignoring))
            applyMacros(line);
          else
            ignoreLine = true;
        }

        if (ignoreLine) {
          linesIgnored.push_back(linePos);
        }
      }

      if (linesIgnored.size() == 0)
        return;

      if (linesIgnored.back() != (allExp.leafCount - 1))
        linesIgnored.push_back(allExp.leafCount);

      const int ignoreCount = (int) linesIgnored.size();

      int start = (linesIgnored[0] + 1);
      int pos   = linesIgnored[0];

      for (int linePos = 1; linePos < ignoreCount; ++linePos) {
        int end = linesIgnored[linePos];

        for (int i = start; i < end; ++i)
          allExp[pos++].value = allExp[i].value;

        start = (end + 1);
      }

      allExp.leafCount = pos;
    }

    expNode parserBase::splitAndPreprocessContent(const std::string &s,
                                                  const int parsingLanguage_) {
      return splitAndPreprocessContent(s.c_str(),
                                       parsingLanguage_);
    }

    expNode parserBase::splitAndPreprocessContent(const char *cRoot,
                                                  const int parsingLanguage_) {
      expNode allExp;

      pushLanguage(parsingLanguage_);

      allExp = splitContent(cRoot, parsingLanguage_);
      // allExp.print();
      // throw 1;

      initModeMacros();

      initMacros(parsingLanguage_);
      preprocessMacros(allExp);

      labelCode(allExp, parsingLanguage_);

      allExp.setNestedSInfo(globalScope);

      popLanguage();

      return allExp;
    }
    //====================================
    void parserBase::initModeMacros() {
      std::string modes[5] = {
        "SERIAL", "OPENMP", "OPENCL", "CUDA", "PTHREADS",
      };
      const std::string currentMode = uppercase(properties["mode"].string());
      for (int i = 0; i < 5; ++i) {
        std::string modeDefine = "#define OCCA_USING_";
        modeDefine += modes[i];
        if (modes[i] != currentMode) {
          modeDefine += " 0";
        } else {
          modeDefine += " 1";
        }
        loadMacro(modeDefine);
      }

      loadMacro("#define OCCA_USING_CPU (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS)");
      loadMacro("#define OCCA_USING_GPU (OCCA_USING_OPENCL || OCCA_USING_CUDA)");
    }

    void parserBase::initMacros(const int parsingLanguage_) {
      if (parsingLanguage_ & parserInfo::parsingFortran)
        initFortranMacros();

      if (macrosAreInitialized)
        return;

      macrosAreInitialized = true;

      //---[ Macros ]---------------------
      loadMacro("#define kernel occaKernel");

      loadMacro("#define barrier(...)   occaBarrier(__VA_ARGS__)");
      loadMacro("#define localMemFence  occaLocalMemFence");
      loadMacro("#define globalMemFence occaGlobalMemFence");

      loadMacro("#define atomicAdd(...)  occaAtomicAdd(__VA_ARGS__)");
      loadMacro("#define atomicSub(...)  occaAtomicSub(__VA_ARGS__)");
      loadMacro("#define atomicSwap(...) occaAtomicSwap(__VA_ARGS__)");
      loadMacro("#define atomicInc(...)  occaAtomicInc(__VA_ARGS__)");
      loadMacro("#define atomicDec(...)  occaAtomicDec(__VA_ARGS__)");
      loadMacro("#define atomicMin(...)  occaAtomicMin(__VA_ARGS__)");
      loadMacro("#define atomicMax(...)  occaAtomicMax(__VA_ARGS__)");
      loadMacro("#define atomicAnd(...)  occaAtomicAnd(__VA_ARGS__)");
      loadMacro("#define atomicOr(...)   occaAtomicOr(__VA_ARGS__)");
      loadMacro("#define atomicXor(...)  occaAtomicXor(__VA_ARGS__)");

      loadMacro("#define atomicAdd64(...)  occaAtomicAdd64(__VA_ARGS__)");
      loadMacro("#define atomicSub64(...)  occaAtomicSub64(__VA_ARGS__)");
      loadMacro("#define atomicSwap64(...) occaAtomicSwap64(__VA_ARGS__)");
      loadMacro("#define atomicInc64(...)  occaAtomicInc64(__VA_ARGS__)");
      loadMacro("#define atomicDec64(...)  occaAtomicDec64(__VA_ARGS__)");

      loadMacro("#define shared   occaShared");
      loadMacro("#define restrict occaRestrict");
      loadMacro("#define volatile occaVolatile");
      loadMacro("#define aligned  occaAligned");
      loadMacro("#define const    occaConst");
      loadMacro("#define constant occaConstant");

      std::string mathFunctions[18] = {
        "min" , "max"  ,
        "sqrt", "sin"  , "asin" ,
        "sinh", "asinh", "cos"  ,
        "acos", "cosh" , "acosh",
        "tan" , "atan" , "tanh" ,
        "atanh", "exp" , "log2" ,
        "log10"
      };

      for (int i = 0; i < 18; ++i) {
        std::string mf = mathFunctions[i];
        std::string cmf = mf;
        cmf[0] += ('A' - 'a');

        loadMacro("#define "       + mf  + "(...) occa"       + cmf + "(__VA_ARGS__)");
        loadMacro("#define fast"   + cmf + "(...) occaFast"   + cmf + "(__VA_ARGS__)");
        loadMacro("#define native" + cmf + "(...) occaNative" + cmf + "(__VA_ARGS__)");
      }

      //---[ CUDA Macros ]--------------
      loadMacro("#define __global__ occaKernel");

      loadMacro("#define __syncthreads()       occaBarrier(occaGlobalMemFence)");
      loadMacro("#define __threadfence_block() occaBarrier(occaLocalMemFence)");
      loadMacro("#define __threadfence()       occaBarrier(occaGlobalMemFence)");

      loadMacro("#define __shared__   occaShared");
      loadMacro("#define __restrict__ occaRestrict");
      loadMacro("#define __volatile__ occaVolatile");
      loadMacro("#define __constant__ occaConstant");

      loadMacro("#define __device__ occaFunction");

      //---[ OpenCL Macros ]------------
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

    void parserBase::loadLanguageTypes() {
      pushLanguage(parserInfo::parsingC);

      cPodTypes["bool"]   = 0;
      cPodTypes["char"]   = 0;
      cPodTypes["short"]  = 0;
      cPodTypes["int"]    = 0;
      cPodTypes["long"]   = 0;
      cPodTypes["float"]  = 0;
      cPodTypes["double"] = 0;

      int parts[6]            = {1, 2, 3, 4, 8, 16};
      std::string suffix[6]   = {"", "2", "3", "4", "8", "16"};
      std::string baseType[8] = {"void",
                                 "int"  ,
                                 "bool" ,
                                 "char" ,
                                 "long" ,
                                 "short",
                                 "float",
                                 "double"};

      std::stringstream ss;

      for (int t = 0; t < 8; ++t) {
        for (int n = 0; n < 6; ++n) {
          typeInfo &type = *(new typeInfo);

          if (n == 0) {
            type.name     = baseType[t] + suffix[n];
            type.baseType = &type;

            globalScope->addType(type);
          }
          else {
            ss << "struct " << baseType[t] << parts[n] << " {\n";

            for (int n2 = 0; n2 < parts[n]; ++n2) {
              const char varLetter = ('w' + ((n2 + 1) % 4));
              const char varNum1   = ((n2 < 10) ? ('0' + n2) : ('a' + (n2 - 10)));
              const char varNum2   = ((n2 < 10) ? ('0' + n2) : ('A' + (n2 - 10)));

              const bool needsUnion = ((n2 < 4) || (10 <= n2));

              std::string tab = (needsUnion ? "    " : "  ");

              if (needsUnion)
                ss << "  union {\n";

              if (n2 < 4)
                ss << tab << baseType[t] << " " << varLetter << ";\n";

              ss << tab << baseType[t] << " s" << varNum1 << ";\n";

              if (10 <= n2)
                ss << tab << baseType[t] << " s" << varNum2 << ";\n";

              if (needsUnion)
                ss << "  };\n";
            }

            ss << "};";

            expNode typeExp = globalScope->createPlainExpNodeFrom(ss.str());

            type.loadFrom(*globalScope, typeExp);
            // typeExp.free(); [<>] Errors out

            globalScope->addType(type);

            ss.str("");
          }

          if (type.name == "void")
            break;
        }
      }

      popLanguage();
    }

    void parserBase::initFortranMacros() {
      if (macrosAreInitialized)
        return;

      macrosAreInitialized = true;
    }

    void parserBase::applyToAllStatements(statement &s,
                                          applyToAllStatements_t func) {
      (this->*func)(s);

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        applyToAllStatements(*(statementPos->value), func);
        statementPos = statementPos->right;
      }
    }

    void parserBase::applyToAllKernels(statement &s,
                                       applyToAllStatements_t func) {
      if (statementIsAKernel(s)) {
        (this->*func)(s);
        return;
      }

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        applyToAllStatements(*(statementPos->value), func);
        statementPos = statementPos->right;
      }
    }

    bool parserBase::statementIsAKernel(statement &s) {
      if (s.info & smntType::functionStatement) {
        if (s.hasQualifier("occaKernel"))
          return true;
      }

      return false;
    }

    statement* parserBase::getStatementKernel(statement &s) {
      if (statementIsAKernel(s))
        return &s;

      statement *sUp = &s;

      while(sUp) {
        if (statementIsAKernel(*sUp))
          return sUp;

        sUp = sUp->up;
      }

      return sUp;
    }

    bool parserBase::statementKernelUsesNativeOCCA(statement &s) {
      statement *sKernel = getStatementKernel(s);

      if (sKernel == NULL)
        return false;

      return sKernel->hasAttribute("__occa__native");
    }

    bool parserBase::statementKernelUsesNativeOKL(statement &s) {
      statement *sKernel = getStatementKernel(s);

      if (sKernel == NULL)
        return false;

      return sKernel->hasAttribute("__okl__native");
    }

    bool parserBase::statementKernelUsesNativeLanguage(statement &s) {
      if (statementKernelUsesNativeOKL(s))
        return true;

      if (statementKernelUsesNativeOCCA(s))
        return true;

      return false;
    }

    void parserBase::setupOccaFors(statement &s) {
      if (compilingForCPU())
        return;

      if (s.info != smntType::occaFor)
        return;

      statement *spKernel = getStatementKernel(s);

      if (spKernel == NULL)
        return;

      if (statementKernelUsesNativeOCCA(s))
        return;

      occaLoopInfo loopInfo(s, parsingLanguage);

      std::string loopTag, capLoopTag, loopNest;
      std::string iter, start;
      std::string bound, iterCheck;
      std::string opStride, opSign, iterOp;

      loopInfo.getLoopInfo(loopTag, loopNest);

      // Capitalized tag
      capLoopTag = loopTag;
      capLoopTag[0] += ('A' - 'a');

      loopInfo.getLoopNode1Info(iter, start);
      loopInfo.getLoopNode2Info(bound, iterCheck);
      loopInfo.getLoopNode3Info(opStride, opSign, iterOp);

      std::string setupExp = loopInfo.getSetupExpression();

      std::string occaIdName  = "occa" + capLoopTag + "Id"  + loopNest;
      std::string occaForName = "occa" + capLoopTag + "For" + loopNest;

      std::string stride = ((opSign[0] == '-') ? "-(" : "(");
      stride += opStride;
      stride += ")";
      //================================

      std::stringstream ss;

      // Working Dim
      ss << "@(occaIterExp = "
         << "("
         <<   "((" << bound << ") - (" << start << ") + (" << stride << " - 1))"
         <<   " / (" << stride << ")"
         << "))";

      s.addAttribute(ss.str());
      ss.str("");

      removeIntraLoopDepsFromIterExp(s);

      std::string setupAttr = "@isAnOccaIterExp";

      if (opStride != "1") {
        ss << setupExp;

        ss << ' '
           << opSign
           << " (" << occaIdName << " * (" << opStride << ")) " << setupAttr << ";";
      }
      else {
        ss << setupExp;

        ss << ' ' << opSign << ' ' << occaIdName << ' ' << setupAttr << ';';
      }

      varInfo &iterVar = *(s.hasVariableInScope(iter));

      varOriginMap[&iterVar] = NULL;

      s.removeFromScope(iterVar);

      s.pushSourceLeftOf(s.statementStart, ss.str());

      s.expRoot.info  = expType::occaFor;
      s.expRoot.value = occaForName;
      s.expRoot.free();

      s.info = smntType::occaFor;

      // Change the origin of the iteration variable
      statement &newIterS = *(s.statementStart->value);
      expNode &initNode   = newIterS.expRoot;

      expNode *newVarNode = ((initNode.info & expType::declaration) ?
                             initNode.getVariableInfoNode(0) :
                             initNode.getUpdatedVariableInfoNode(0));

      if (newVarNode) {
        newVarNode->freeThis();
        newVarNode->putVarInfo(iterVar);

        newVarNode->info |= expType::type;

        newIterS.addVariable(&iterVar, newIterS.up);
      }
    }

    void parserBase::removeIntraLoopDepsFromIterExp(statement &s) {
      expNode &occaIterExp = s.attribute("occaIterExp").valueExp();
      expNode &flatRoot = *(occaIterExp.makeFlatHandle());

      varOriginMap_t deps;

      findDependenciesFor(occaIterExp, deps);

      varOriginMapIterator it = deps.begin();

      while(it != deps.end()) {
        varInfo   &var       = *(it->first);
        statement &varOrigin = *(it->second);

        if (!varOrigin.hasAttribute("isAnOccaIterExp")) {
          ++it;
          continue;
        }

        expNode &initNode = *(varOrigin.getDeclarationVarInitNode(0));

        for (int i = 0; i < flatRoot.leafCount; ++i) {
          expNode &leaf = flatRoot[i];

          // [---] Temporary fix
          if (((leaf.info & expType::varInfo) == 0) ||
             (leaf.getVarInfo().name != var.name)) {

            continue;
          }

          expNode &leafUp   = *(leaf.up);
          const int leafPos = leaf.whichLeafAmI();

          // leaf.free();

          expNode &newLeaf = *(initNode.clonePtr());
          zeroOccaIdsFrom(newLeaf);

          leafUp.setLeaf(newLeaf, leafPos);
        }

        ++it;
      }
    }

    bool parserBase::statementIsOccaOuterFor(statement &s) {
      if (s.info == smntType::occaFor) {
        std::string &forName = s.expRoot.value;

        if ((forName.find("occaOuterFor") != std::string::npos) &&
           ((forName == "occaOuterFor0") ||
            (forName == "occaOuterFor1") ||
            (forName == "occaOuterFor2"))) {

          return true;
        }
      }

      return false;
    }

    bool parserBase::statementIsOccaInnerFor(statement &s) {
      if (s.info == smntType::occaFor) {
        std::string &forName = s.expRoot.value;

        if ((forName.find("occaInnerFor") != std::string::npos) &&
           ((forName == "occaInnerFor0") ||
            (forName == "occaInnerFor1") ||
            (forName == "occaInnerFor2"))) {

          return true;
        }
      }

      return false;
    }

    bool parserBase::statementHasOccaOuterFor(statement &s) {
      if (statementIsOccaOuterFor(s))
        return true;

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        if ( statementHasOccaOuterFor(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    bool parserBase::statementHasOccaFor(statement &s) {
      if ((s.info == smntType::occaFor) &&
         (s.getForStatementCount() == 0)) {

        return true;
      }

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        if ( statementHasOccaFor(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    bool parserBase::statementHasOklFor(statement &s) {
      if ((s.info == smntType::occaFor) &&
         (0 < s.getForStatementCount())) {

        return true;
      }

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        if ( statementHasOklFor(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    bool parserBase::statementHasOccaStuff(statement &s) {
      if (statementHasOklFor(s))
        return true;

      if (statementHasOccaOuterFor(s))
        return true;

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        if ( statementHasOccaStuff(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    void parserBase::reorderLoops() {
      statementVector_t loopsToReorder; // ltr

      placeLoopsToReorder(*globalScope, loopsToReorder);

      const int ltrCount = (int) loopsToReorder.size();

      if (ltrCount < 2)
        return;

      int start = 0;
      statement *sRoot = loopsToReorder[0];

      // depMap_t depMap(*sRoot);

      // varInfo &ai        = *(sRoot->hasVariableInScope("ai"));
      // varDepInfo &vdInfo = depMap(ai);

      // std::cout << "vdInfo = " << vdInfo << '\n';

      for (int i = 1; i < ltrCount; ++i) {
        statement &s = *(loopsToReorder[i]);

        if (!s.insideOf(*sRoot)) {
          if ((i - start) < 2) {
            start = i;
            sRoot = &s;

            continue;
          }

          reorderLoops(loopsToReorder, start, i);

          start = i;
          sRoot = &s;
        }
      }

      if (2 <= (ltrCount - start))
        reorderLoops(loopsToReorder, start, ltrCount);
    }

    void parserBase::reorderLoops(statementVector_t &loopsToReorder,
                                  const int start,
                                  const int end) {

      if ((end - start) < 2)
        return;

      intVector_t relatedLoops = relatedReorderLoops(loopsToReorder,
                                                     start,
                                                     end);

      const int rlCount = (int) relatedLoops.size();

      intVector_t oldTo(rlCount);
      intVector_t newFrom(rlCount);

      for (int i = 0; i < rlCount; ++i) {
        statement &s      = *(loopsToReorder[relatedLoops[i]]);
        attribute_t &attr = *(s.hasAttribute("loopOrder"));

        s.removeAttribute("loopOrder");

        const int loopPos   = atoi(attr[attr.argCount - 1].value);
        oldTo[i]            = (rlCount - loopPos - 1);
        newFrom[ oldTo[i] ] = i;
      }

      for (int i = 0; i < rlCount; ++i) {
        const int to   = i;
        const int from = newFrom[i];

        if (from == to)
          continue;

        newFrom[ oldTo[i] ] = from;
        oldTo[ from ]       = oldTo[i];

        statement::swapStatementNodesFor(*(loopsToReorder[relatedLoops[to]]),
                                         *(loopsToReorder[relatedLoops[from]]));

        swapValues(relatedLoops[to], relatedLoops[from]);
      }

      for (int i = start; i < end; ++i) {
        if (loopsToReorder[i]->hasAttribute("loopOrder"))
          reorderLoops(loopsToReorder, i, end);
      }
    }

    intVector_t parserBase::relatedReorderLoops(statementVector_t &loopsToReorder,
                                                const int start,
                                                const int end) {

      intVector_t relatedLoops;
      relatedLoops.push_back(start);

      statement &sRoot      = *(loopsToReorder[start]);
      attribute_t &rootAttr = sRoot.attribute("loopOrder");

      for (int i = (start + 1); i < end; ++i) {
        statement &s       = *(loopsToReorder[i]);
        attribute_t *attr_ = s.hasAttribute("loopOrder");

        // This statement is already taken
        if (attr_ == NULL)
          continue;

        attribute_t &attr = *(attr_);

        if (attr.argCount != rootAttr.argCount)
          continue;

        if (rootAttr.argCount == 1) {
          relatedLoops.push_back(i);
        }
        else if ((rootAttr.argCount == 2) &&
                (rootAttr[0].value == attr[0].value)) {

          relatedLoops.push_back(i);
        }
      }

      return relatedLoops;
    }

    void parserBase::placeLoopsToReorder(statement &s,
                                         statementVector_t &loopsToReorder) {

      if ((s.info & smntType::forStatement) &&
         s.hasAttribute("loopOrder")) {

        loopsToReorder.push_back(&s);
      }

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        placeLoopsToReorder(*(statementPos->value),
                            loopsToReorder);

        statementPos = statementPos->right;
      }
    }

    void parserBase::retagOccaLoops() {
      retagOccaLoops(*globalScope);
    }

    void parserBase::retagOccaLoops(statement &s) {
      if (s.info == smntType::occaFor) {
        statementVector_t occaLoops = findOccaLoops(s);
        const int occaLoopCount     = (int) occaLoops.size();

        int outerCount = 0;

        // Count outer/inner loops
        for (int i = 0; i < occaLoopCount; ++i) {
          statement &occaLoop = *(occaLoops[i]);

          attribute_t &occaTagAttr = occaLoop.attribute("occaTag");
          std::string occaTag      = occaTagAttr.valueStr();

          if (occaTag == "tile") {
            return;
          }
          if (occaTag == "outer") {
            ++outerCount;
          }
        }

        // Reorder tags
        for (int i = 0; i < occaLoopCount; ++i) {
          statement &occaLoop = *(occaLoops[i]);

          occaLoop.removeAttribute("occaTag");
          occaLoop.removeAttribute("occaNest");
          occaLoop.removeAttribute("occaMaxNest_inner");
          occaLoop.removeAttribute("occaMaxNest_outer");

          if (outerCount) {
            std::string nestStr = occa::toString(--outerCount);

            occaLoop.addAttribute("@(occaTag = outer,"
                                  " occaNest = " + nestStr + ")");
          }
          else {
            // innerCount includes itself
            const int innerCount = (findOccaLoops(occaLoop).size() - 1);
            std::string nestStr = occa::toString(innerCount);

            occaLoop.addAttribute("@(occaTag = inner,"
                                  " occaNest = " + nestStr + ")");
          }
        }

        // Update max outer/inner dims
        for (int i = 0; i < occaLoopCount; ++i) {
          statement &occaLoop = *(occaLoops[i]);

          attribute_t &occaTagAttr  = occaLoop.attribute("occaTag");
          attribute_t &occaNestAttr = occaLoop.attribute("occaNest");

          occaLoop.updateOccaOMLoopAttributes(occaTagAttr.valueStr(),
                                              occaNestAttr.valueStr());
        }

        return;
      }

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        retagOccaLoops(*(statementPos->value));

        statementPos = statementPos->right;
      }
    }

    void parserBase::splitTileOccaFors(statement &s) {
      if (s.info != smntType::occaFor)
        return;

      attribute_t *occaTagAttr_ = s.hasAttribute("occaTag");

      if ((occaTagAttr_ == NULL) ||
         (occaTagAttr_->valueStr() != "tile")) {

        return;
      }

      attribute_t &occaTagDim = s.attribute("tileDim");

      expNode &initNode   = *(s.getForStatement(0));
      expNode &checkNode  = *(s.getForStatement(1));
      expNode &updateNode = *(s.getForStatement(2));

      expNode &csvCheckNode  = *(checkNode.makeCsvFlatHandle());
      expNode &csvUpdateNode = *(updateNode.makeCsvFlatHandle());

      //---[ Checks ]---------------------------
      //  ---[ Tile Dim ]-------------
      const int tileDim = occaTagDim.argCount;

      OCCA_ERROR("Only 1D, 2D, and 3D tiling are supported:\n" << s.onlyThisToString(),
                 (1 <= tileDim) && (tileDim <= 3));

      int varsInInit = ((initNode.info & expType::declaration) ?
                        initNode.getVariableCount()            :
                        initNode.getUpdatedVariableCount());

      OCCA_ERROR("Only one iterator can be initialized:\n" << s.onlyThisToString(),
                 varsInInit == 1);

      expNode *varInitNode = ((initNode.info & expType::declaration) ?
                              initNode.getVariableInitNode(0)        :
                              initNode.getUpdatedVariableSetNode(0));

      expNode *csvInitValueNode_;

      if (tileDim == 1) {
        csvInitValueNode_ = varInitNode->makeCsvFlatHandle();
      }
      else {
        OCCA_ERROR("Iterator is not defined properly (e.g. int2 i = {0,0}):\n" << s.onlyThisToString(),
                   varInitNode->value == "{");

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

      if (1 < tileDim) {
        suffix = '0';
        suffix[0] += tileDim;
      }

      if (     varTypeN == ("int"   + suffix)) varType = "int";
      else if (varTypeN == ("char"  + suffix)) varType = "char";
      else if (varTypeN == ("long"  + suffix)) varType = "long";
      else if (varTypeN == ("short" + suffix)) varType = "short";

      OCCA_ERROR("Iterator [" << var << "] is not a proper type (e.g. int" << suffix << ')',
                 0 < varType.size());

      //  ---[ Proper check vars ]----
      int varsInCheck = csvCheckNode.leafCount;

      OCCA_ERROR("Only one variable can be checked:\n" << s.onlyThisToString(),
                 varsInCheck == tileDim);

      expNode **orderBuffer = new expNode*[csvCheckNode.leafCount];
      bool *checkIterOnLeft = new bool[csvCheckNode.leafCount];

      for (int dim = 0; dim < tileDim; ++dim)
        orderBuffer[dim] = NULL;

      for (int dim = 0; dim < tileDim; ++dim) {
        expNode &check = csvCheckNode[dim];
        int dim2 = dim;

        OCCA_ERROR("Error on: " << s.onlyThisToString() << "\n\n"
                   << "Check operator must be in [<=, <, >, >=]: " << check.toString(),
                   (check.info == expType::LR) &&
                   ((check.value == "<=") ||
                    (check.value == "<" ) ||
                    (check.value == ">" ) ||
                    (check.value == ">=")));

        int side;

        for (side = 0; side < 2; ++side) {
          if (tileDim == 1) {
            if ((check[side].value == var.name)) {
              checkIterOnLeft[dim2] = (side == 0);

              break;
            }
          }
          else {
            if ((check[side].value    == ".") &&
               (check[side][0].value == var.name)) {

              dim2 = (check[side][1].value[0] - 'x');
              checkIterOnLeft[dim2] = (side == 0);

              break;
            }
          }
        }

        OCCA_ERROR("Error on: " << s.onlyThisToString() << "\n\n"
                   << "Variable checks must look like:\n"
                   "  X op Y where op can be [<=, <, >, >=]\n"
                   "  X or Y must be for-loop iterator\n"
                   "  For 2D or 3D tiling: X.x < Y, X.y < Y, X.z < Y (order doesn't matter)",
                   side < 2);

        orderBuffer[dim2] = &(csvCheckNode[dim]);
      }

      for (int dim = 0; dim < tileDim; ++dim) {
        OCCA_ERROR(var.name << '.' << (char) ('x' + dim) << " needs to be checked: " << s.onlyThisToString(),
                   orderBuffer[dim] != NULL);


        csvCheckNode.leaves[dim] = orderBuffer[dim];
        orderBuffer[dim]         = NULL;
      }

      //  ---[ Proper update vars ]---
      int varsInUpdate = csvUpdateNode.leafCount;

      OCCA_ERROR("Only one variable can be updated:\n" << s.onlyThisToString(),
                 varsInUpdate == tileDim);

      for (int dim = 0; dim < tileDim; ++dim) {
        expNode &update = csvUpdateNode[dim];
        int dim2 = dim;

        OCCA_ERROR("Update operator must be in [++, --, +=, -=]: " << update.toString(),
                   (update.value == "++") ||
                   (update.value == "--") ||
                   (update.value == "+=") ||
                   (update.value == "-="));

        if (1 < tileDim) {
          OCCA_ERROR("Iterator [" << var.name << "] is not updated, [" << update[0][0].value << "] is updated instead",
                     update[0][0].value == var.name);

          dim2 = (update[0][1].value[0] - 'x');
        }

        orderBuffer[dim2] = &(csvUpdateNode[dim]);
      }

      for (int dim = 0; dim < tileDim; ++dim) {
        OCCA_ERROR(var.name << '.' << (char) ('x' + dim) << " needs to be updated: " << s.onlyThisToString(),
                   orderBuffer[dim] != NULL);

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

      for (int dim = (tileDim - 2); 0 <= dim; --dim) {
        oStatements[dim] = oStatements[dim + 1]->makeSubStatement();
        oStatements[dim + 1]->addStatement(oStatements[dim]);
      }

      iStatements[tileDim - 1] = oStatements[0]->makeSubStatement();
      oStatements[0]->addStatement(iStatements[tileDim - 1]);

      for (int dim = (tileDim - 2); 0 <= dim; --dim) {
        iStatements[dim] = iStatements[dim + 1]->makeSubStatement();
        iStatements[dim + 1]->addStatement(iStatements[dim]);
      }

      // Place s's statementNode's in inner-most for-loop
      iStatements[0]->statementStart = s.statementStart;
      iStatements[0]->statementEnd   = s.statementEnd;

      statementNode *sn = iStatements[0]->statementStart;

      while(sn) {
        sn->value->up = iStatements[0];
        sn = sn->right;
      }

      std::stringstream ss;

      for (int dim = 0; dim < tileDim; ++dim) {
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

        ss << "for ("
           << varType << ' ' << oTileVar << " = " << csvInitValueNode[dim].toString() << "; ";

        if (checkIterOnLeft[dim])
          ss << oTileVar << check.value << check[1].toString() << "; ";
        else
          ss << check[0].toString() << check.value << oTileVar << "; ";

        if (update.info != expType::LR) {
          if (update.value == "++")
            ss << oTileVar << " += " << occaTagDim[dim] << "; ";
          else
            ss << oTileVar << " -= " << occaTagDim[dim] << "; ";
        }
        else {
          ss << oTileVar << update.value << occaTagDim[dim] << "; ";
        }

        ss << "outer" << dim << ')';

        std::string outerForSource = ss.str();

        ss.str("");

        std::string varName = var.name;

        if (1 < tileDim)
          varName = iTileVar;

        ss << "for (";

        ss << varType << ' '
           << varName << " = " << oTileVar << "; ";

        if (checkIterOnLeft[dim])
          ss << varName << check.value << '(' << oTileVar << " + " << occaTagDim[dim] << "); ";
        else
          ss << '(' << oTileVar << " + " << occaTagDim[dim] << ')' << check.value << varName << "; ";

        csvUpdateNode[dim][0].free();
        csvUpdateNode[dim][0].info  = expType::printValue;
        csvUpdateNode[dim][0].value = varName;

        ss << csvUpdateNode[dim].toString() << "; ";

        ss << "inner" << dim << ')';

        std::string innerForSource = ss.str();

        ss.str("");

        os.expRoot.free();
        is.expRoot.free();

        if ((dim == 0) &&
           (varIsDeclared)) {

          varOriginMap[&var] = NULL;

          iStatements[0]->removeFromScope(var);
        }

        os.reloadFromSource(outerForSource);
        is.reloadFromSource(innerForSource);

        // [--] Nasty, fix later
        if ((dim == 0) &&
           (varIsDeclared)) {

          varOriginMap[&var] = &is;
        }

        os.updateOccaOMLoopAttributes("outer", toString(dim));
        is.updateOccaOMLoopAttributes("inner", toString(dim));
      }

      // Add variable declaration if needed
      if (tileDim == 1) {
        if (varIsDeclared) {
          expNode &newInitNode = *(iStatements[0]->getForStatement(0));
          expNode &varNode     = *(newInitNode.getVariableInfoNode(0));

          varNode.freeThis();
          varNode.putVarInfo(var);

          varNode.info |= expType::type;
        }
      }
      else { // (1 < tileDim)
        statement &is = *(iStatements[0]);

        if (varIsDeclared)
          is.pushSourceLeftOf(is.statementStart,
                              (std::string) var + ";");

        statementNode *sn2 = is.statementStart;

        for (int dim = 0; dim < tileDim; ++dim) {
          ss << var.name << "." << (char) ('x' + dim) << " = __occa_iTileVar" << dim << ';';

          is.pushSourceRightOf(sn2,
                               ss.str());

          ss.str("");

          sn2 = sn2->right;
        }
      }

      expNode::freeFlatHandle(csvCheckNode);
      expNode::freeFlatHandle(csvInitValueNode);
      expNode::freeFlatHandle(csvUpdateNode);

      delete [] checkIterOnLeft;
    }

    void parserBase::markKernelFunctions() {
      statementNode *snPos = globalScope->statementStart;

      while(snPos) {
        statement &s2 = *(snPos->value);

        if ( !(s2.info & smntType::functionStatement) ||
            statementIsAKernel(s2) ) {

          snPos = snPos->right;
          continue;
        }

        if (statementHasOccaStuff(s2)) {
          varInfo &fVar = *(s2.getFunctionVar());
          fVar.addQualifier("occaKernel", 0);
        }

        snPos = snPos->right;
      }
    }

    void parserBase::labelNativeKernels() {
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos) {
        statement &s = *(statementPos->value);

        if (statementIsAKernel(s) && // Kernel
           (s.statementStart != NULL)) {

          bool hasOccaFor = statementHasOccaFor(s);
          bool hasOklFor  = statementHasOklFor(s);

          if (hasOccaFor | hasOklFor) {
            if (hasOccaFor) {
              if (!s.hasAttribute("__occa__native"))
                s.addAttributeTag("__occa__native");
            }
            else {
              if (!s.hasAttribute("__okl__native"))
                s.addAttributeTag("__okl__native");
            }
          }
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::setupCudaVariables(statement &s) {
      if ((!(s.info & smntType::simpleStatement)    &&
          !(s.info & smntType::forStatement)       &&
          !(s.info & smntType::functionStatement)) ||
         // OCCA for's don't have arguments
         (s.info == smntType::occaFor))
        return;

      if (getStatementKernel(s) == NULL)
        return;

      // [-] Go Franken-kernels ...
      // if (statementKernelUsesNativeLanguage(s))
      //   return;

      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        std::string &value = flatRoot[i].value;
        std::string occaValue;

        bool compressing = false;

        if (value == "threadIdx") {
          compressing = true;
          occaValue = "occaInnerId";
        }
        else if (value == "blockDim") {
          compressing = true;
          occaValue = "occaInnerDim";
        }
        else if (value == "blockIdx") {
          compressing = true;
          occaValue = "occaOuterId";
        }
        else if (value == "gridDim") {
          compressing = true;
          occaValue = "occaOuterDim";
        }

        if (compressing) {
          expNode &leaf    = *(flatRoot[i].up);
          const char coord = (leaf[1].value[0] + ('0' - 'x'));

          leaf.info  = (expType::presetValue | expType::occaKeyword);
          leaf.value = occaValue + coord;

          leaf.leafCount = 0;
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void parserBase::addFunctionPrototypes() {
      std::map<std::string,bool> prototypes;

      statementNode *statementPos = globalScope->statementStart;

      while(statementPos) {
        statement &s = *(statementPos->value);

        if (s.info & smntType::functionPrototype)
          prototypes[s.getFunctionName()] = true;

        statementPos = statementPos->right;
      }

      statementPos = globalScope->statementStart;

      while(statementPos) {
        statement &s = *(statementPos->value);

        if (s.info & smntType::functionStatement) {
          if (s.hasQualifier("occaKernel")) {
            statementPos = statementPos->right;
            continue;
          }

          if (!s.hasQualifier("occaFunction"))
            s.addQualifier("occaFunction");

          if ( !(s.info & smntType::functionDefinition) ) {
            statementPos = statementPos->right;
            continue;
          }

          if (prototypes.find( s.getFunctionName() ) == prototypes.end()) {
            globalScope->pushSourceLeftOf(statementPos,
                                          (std::string) *(s.getFunctionVar()));
          }
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::checkOccaBarriers(statement &s) {
      if (!warnForConditionalBarriers())
        return;

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        statement &s2 = *(statementPos->value);

        if (s2.info & smntType::ifStatement) {
          if (s2.hasStatementWithBarrier()) {
            OCCA_ERROR("Barriers are not allowed in conditional statements:\n" << s2,
                       false);
          }
        }
        else
          checkOccaBarriers(s2);

        statementPos = statementPos->right;
      }
    }

    void parserBase::addOccaBarriers() {
      if (!insertBarriersAutomatically()) {
        return;
      }

      statementNode *statementPos = globalScope->statementStart;

      while(statementPos) {
        statement &s = *(statementPos->value);

        if (statementIsAKernel(s)) {
          statementVector_t loops;
          findInnerLoopSets(s, loops);
          const int loopCount = (int) loops.size();
          for (int i = 0; i < (loopCount - 1); ++i) {
            statement &s1 = *(loops[i]);
            statement &s2 = *(loops[i + 1]);
            if (statementUsesShared(s2) &&
                !barrierBetween(s1, s2)) {
              s2.up->pushSourceLeftOf(s2.getStatementNode(),
                                      "occaBarrier(occaLocalMemFence);");
            }
          }
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::findInnerLoopSets(statement &s, statementVector_t &loops) {
      statementNode *statementPos = s.statementStart;

      while (statementPos) {
        statement &s2 = *(statementPos->value);

        attribute_t *occaTagAttr = s2.hasAttribute("occaTag");

        if (occaTagAttr &&
            occaTagAttr->valueStr() == "inner") {
          loops.push_back(&s2);
        } else {
          findInnerLoopSets(s2, loops);
        }

        statementPos = statementPos->right;
      }
    }

    bool parserBase::statementUsesShared(statement &s) {
      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        if ((flatRoot[i].info & expType::varInfo) &&
            flatRoot[i].getVarInfo().hasQualifier("occaShared")) {

          expNode::freeFlatHandle(flatRoot);
          return true;
        }
      }

      expNode::freeFlatHandle(flatRoot);

      statementNode *statementPos = s.statementStart;
      while (statementPos) {
        statement &s2 = *(statementPos->value);
        if (statementUsesShared(s2)) {
          return true;
        }
        statementPos = statementPos->right;
      }

      return false;
    }

    bool parserBase::barrierBetween(statement &s1, statement &s2) {
      statement &gcs = s1.greatestCommonStatement(s2);

      statementVector_t path[2];
      statementNodeVector_t nodes[2];

      for (int pass = 0; pass < 2; ++pass) {
        statement *cs = ((pass == 0) ? &s1 : &s2);

        // Get path leading to the GCS
        while (cs != &gcs) {
          path[pass].push_back(cs);
          nodes[pass].push_back(cs->getStatementNode());
          cs = cs->up;
        }

        // Check all following child statements in each parent (except GCS)
        const int pathSize = (int) path[pass].size();
        for (int i = 0; i < (pathSize - 1); ++i) {
          // We only want to check
          //   - s1's parent's rights
          //   - s2's parent's lefts
          // Otherwise we might get statements left of s1 or right of s2
          if (barrierBetween(pass ? nodes[pass][i]->left : nodes[pass][i]->right,
                             NULL)) {
            return true;
          }
        }
      }

      // Search nodes between GCS childs leading to s1 and s2
      return barrierBetween(nodes[0][path[0].size() - 1]->right,
                            nodes[1][path[1].size() - 1]);
    }

    bool parserBase::barrierBetween(statementNode *sn1, statementNode *sn2) {
      while (sn1 && (sn1 != sn2)) {
        statement &s = *(sn1->value);
        if (s.hasStatementWithBarrier()) {
          return true;
        }
        sn1 = sn1->right;
      }
      return false;
    }

    void parserBase::updateConstToConstant() {
      statementNode *snPos = globalScope->statementStart;

      while(snPos) {
        statement &s = *(snPos->value);

        if ((s.info & smntType::declareStatement) &&
           (s.hasQualifier("occaConst"))) {

          s.removeQualifier("occaConst");
          s.addQualifier("occaConstant");
        }

        snPos = snPos->right;
      }
    }

    void parserBase::addArgQualifiers() {
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos) {
        statement &s = *(statementPos->value);

        if ((s.info & smntType::functionDefinition) &&
           (s.functionHasQualifier("occaKernel"))) {

          addArgQualifiersTo(s);
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::addArgQualifiersTo(statement &s) {
      const int argc = s.getFunctionArgCount();

      for (int i = 0; i < argc; ++i) {
        varInfo &argVar = *(s.getFunctionArgVar(i));

        if (argVar.name == "occaKernelInfoArg")
          continue;

        if (argVar.pointerDepth()) {
          if (!argVar.hasQualifier("occaPointer"))
            argVar.addQualifier("occaPointer", 0);
        }
        else {
          if (!argVar.hasQualifier("occaConst"))
            argVar.addQualifier("occaConst");

          if (!argVar.hasRightQualifier("occaVariable"))
            argVar.addRightQualifier("occaVariable");

          if (argVar.hasRightQualifier("&"))
            argVar.removeRightQualifier("&");
        }
      }

      if ((s.getFunctionArgCount() == 0) ||
         (s.getFunctionArgName(0) != "occaKernelInfoArg")) {

        varInfo &arg0 = *(new varInfo());

        arg0.name = "occaKernelInfoArg";

        s.addFunctionArg(0, arg0);
      }
    }

    void parserBase::floatSharedAndExclusivesUp(statement &s) {
      statementNode sn;

      // Get all shared and exclusive variables inside inner-loops
      appendSharedAndExclusives(s, &sn);

      statementNode *statementPos = sn.right;

      while(statementPos) {
        statement &s2  = *(statementPos->value);
        statement *sUp = s2.up;

        // We're moving the definition else-where
        if (sUp) {
          varInfo &var = s2.getDeclarationVarInfo(0);

          sUp->removeVarFromScope(var.name);
          sUp->removeStatement(s2);
        }

        // Find inner-most outer-for loop
        while(sUp) {
          if ((sUp->info == smntType::occaFor) &&
             statementIsOccaOuterFor(*sUp)) {
            break;
          }

          sUp = sUp->up;
        }

        if (sUp) {
          statementNode *sn3 = sUp->statementStart;

          // Skip exclusive and shared statements
          while(sn3) {
            statement &s3 = *(sn3->value);

            if ((!(s3.info & smntType::declareStatement)) ||
               (!s3.hasQualifier("exclusive") &&
                !s3.hasQualifier("occaShared"))) {
              break;
            }

            sn3 = sn3->right;
          }

          const bool appendToEnd   = (sn3 == NULL);
          const bool appendToStart = (!appendToEnd) && (sn3->left == NULL);

          statementNode *sn2 = new statementNode(&s2);

          if (appendToStart) {
            sn2->right                = sUp->statementStart;
            sUp->statementStart->left = sn2;

            sUp->statementStart = sn2;
          }
          else if (appendToEnd) {
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
                                                         bool isAppending) {

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        statement &s2 = *(statementPos->value);

        if ((s2.info & smntType::declareStatement)) {
          if (isAppending &&
             (s2.hasQualifier("exclusive") ||
              s2.hasQualifier("occaShared"))) {

            snTail = snTail->push(&s2);
          }
        }
        else {
          if (statementIsOccaInnerFor(s2))
            isAppending = true;

          snTail = appendSharedAndExclusives(s2, snTail, isAppending);
        }

        statementPos = statementPos->right;
      }

      return snTail;
    }

    void parserBase::modifyExclusiveVariables(statement &s) {
      if ( !(s.info & smntType::declareStatement)   ||
          (getStatementKernel(s) == NULL)    ||
          (statementKernelUsesNativeOCCA(s)) ||
          (!s.hasQualifier("exclusive")) ) {

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

      for (int i = 0; i < argc; ++i) {
        keepVar[i] = newRoot.variableHasInit(i);

        if (keepVar[i])
          ++varsKept;
      }

      if (varsKept) {
        int pos = 0;

        for (int i = 0; i < argc; ++i) {
          if (keepVar[i]) {
            varInfo &newVar = newRoot.getVariableInfoNode(i)->getVarInfo();

            newVar.rightQualifiers.clear();
            newVar.removeStackPointers();

            if (pos != i)
              newRoot.leaves[pos] = newRoot.leaves[i];

            ++pos;
          }
        }

        newRoot.leafCount = varsKept;
      }
      else {
        newRoot.free();
      }
      //================================

      for (int i = 0; i < argc; ++i) {
        varInfo &var = s.getDeclarationVarInfo(i);
        var.removeQualifier("exclusive");

        if (var.hasQualifier("occaConst"))
          var.removeQualifier("occaConst");

        const int isPrivateArray = var.stackPointerCount;

        ss << "occaPrivate";

        if (isPrivateArray)
          ss << "Array";

        ss << "("
           << var.leftQualifiers
           << var.baseType->name
           << var.rightQualifiers << ", "
           << var.name;

        if (isPrivateArray) {
          ss << ", ";

          // [-] Only supports 1D arrays
          OCCA_ERROR("Only 1D exclusive arrays are supported:\n"
                     << "exclusive " << s,
                     var.stackPointerCount < 2);

          ss << var.stackExpRoots[0];
        }

        ss << ");";

        s.loadFromNode( splitAndLabelContent(ss.str()) );

        s.statementEnd->value->up = s.up;

        ss.str("");
      }

      statementNode *sn = s.getStatementNode();

      if (s.up->statementStart != sn) {
        sn->left->right        = s.statementStart;
        s.statementStart->left = sn->left;
      }
      else
        s.up->statementStart = s.statementStart;

      s.statementEnd->right = sn->right;

      if (sn->right)
        sn->right->left = s.statementEnd;

      s.statementStart = s.statementEnd = NULL;
    }

    // [-] Missing
    void parserBase::modifyTextureVariables() {
      /*

        kernel void kern(texture float **tex) {
        tex[j][i];
        }

        CPU:
        kernel void kern(int64 offsets[argc],
        texture float tex,
        sampler/textureInfo tex_info)
      */
    }

    void parserBase::loadKernelInfos() {
      statementNode *snPos = globalScope->statementStart;

      while(snPos) {
        statement &s = *(snPos->value);

        if (statementIsAKernel(s)) {
          snPos = splitKernelStatement(snPos);
        } else {
          snPos = snPos->right;
        }
      }
    }

    statementNode* parserBase::splitKernelStatement(statementNode *snKernel) {

      statement &sKernel       = *(snKernel->value);
      statementNode *lastNewSN = snKernel->right;

      // [O]uter [M]ost [Loops]
      statementVector_t omLoops = findOuterLoopSets(sKernel);
      int kernelCount = (int) omLoops.size();

      statementVector_t newKernels;

      // Add parallel for's
      if (compilingForCPU()) {
        for (int k = 0; k < kernelCount; ++k) {
          statement &omLoop  = *(omLoops[k]);

          sKernel.pushSourceLeftOf(omLoop.getStatementNode(),
                                   "occaParallelFor0");
        }

        kernelCount = 0;
      }

      if (kernelCount) {
        varOriginMapVector_t varDeps(kernelCount);

        for (int k = 0; k < kernelCount; ++k)
          varDeps[k] = findKernelDependenciesFor(sKernel,
                                                 *(omLoops[k]));

        newKernels = newKernelsFromLoops(sKernel,
                                         omLoops,
                                         varDeps);

        addNestedKernelArgTo(sKernel);

        applyToAllStatements(sKernel, &parserBase::zeroOccaIdsFrom);

        for (int k = (kernelCount - 1); 0 <= k; --k)
          sKernel.up->pushRightOf(&sKernel, newKernels[k]);

        lastNewSN = newKernels[kernelCount - 1]->getStatementNode();
      }

      sKernel.up->pushSourceLeftOf(snKernel, "#ifdef OCCA_LAUNCH_KERNEL");

      if (kernelCount) {
        sKernel.up->pushSourceRightOf(snKernel , "#else");
        sKernel.up->pushSourceRightOf(lastNewSN, "#endif");
      } else {
        sKernel.up->pushSourceRightOf(snKernel, "#endif");
        sKernel.up->pushSourceRightOf(snKernel, "#else");

        if (lastNewSN) {
          lastNewSN = lastNewSN->left;
        }
      }

      storeKernelInfo(sKernel, newKernels);

      return ((lastNewSN != NULL) ?
              lastNewSN->right    :
              NULL);
    }

    statementVector_t parserBase::findOuterLoopSets(statement &sKernel) {
      statementVector_t omLoops;

      findOuterLoopSets(sKernel, omLoops);

      return omLoops;
    }

    void parserBase::findOuterLoopSets(statement &s,
                                       statementVector_t &omLoops) {

      if (s.info == smntType::occaFor) {
        attribute_t &occaTagAttr = s.attribute("occaTag");

        if (occaTagAttr.valueStr() == "outer") {
          omLoops.push_back(&s);
          return;
        }
      }

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        statement &s2 = *(statementPos->value);

        findOuterLoopSets(s2, omLoops);

        statementPos = statementPos->right;
      }
    }

    statementVector_t parserBase::findOccaLoops(statement &sKernel) {
      statementVector_t occaLoops;

      findOccaLoops(sKernel, occaLoops);

      return occaLoops;
    }

    void parserBase::findOccaLoops(statement &s,
                                   statementVector_t &occaLoops) {

      if (s.info == smntType::occaFor)
        occaLoops.push_back(&s);

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        statement &s2 = *(statementPos->value);

        findOccaLoops(s2, occaLoops);

        statementPos = statementPos->right;
      }
    }

    varOriginMap_t parserBase::findKernelDependenciesFor(statement &sKernel,
                                                         statement &omLoop) {
      varInfoVector_t depsIgnored;
      varOriginMap_t deps;

      findDependenciesFor(omLoop, deps);

      varOriginMapIterator it = deps.begin();

      while(it != deps.end()) {
        statement &varOrigin = *(it->second);

        if ((&varOrigin   == &sKernel)    ||
           (varOrigin.up == globalScope) ||
           (&varOrigin   == &omLoop)     ||
           (varOrigin.insideOf(omLoop))) {

          depsIgnored.push_back(it->first);
        }

        ++it;
      }

      const int ignoredCount = depsIgnored.size();

      for (int i = 0; i < ignoredCount; ++i)
        deps.erase(depsIgnored[i]);

      return deps;
    }

    varOriginMap_t parserBase::findDependenciesFor(statement &s,
                                                   const int flags) {
      varOriginMap_t deps;

      findDependenciesFor(s, deps, flags);

      return deps;
    }

    void parserBase::findDependenciesFor(statement &s,
                                         varOriginMap_t &deps,
                                         const int flags) {

      findDependenciesFor(s.expRoot, deps);

      if ((flags & parserInfo::checkSubStatements) == 0)
        return;

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        statement &s2 = *(statementPos->value);

        findDependenciesFor(s2, deps, flags);

        statementPos = statementPos->right;
      }
    }

    varOriginMap_t parserBase::findDependenciesFor(expNode &e) {
      varOriginMap_t deps;

      findDependenciesFor(e, deps);

      return deps;
    }

    void parserBase::findDependenciesFor(expNode &e,
                                         varOriginMap_t &deps) {
      expNode &flatRoot = *(e.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        if (flatRoot[i].info & expType::varInfo) {
          varInfo &var = flatRoot[i].getVarInfo();

          // varInfo is also used for casts
          if (var.name.size() == 0)
            continue;

          varOriginMapIterator it = deps.find(&var);

          if (it != deps.end())
            continue;

          deps[&var] = varOriginMap[&var];
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    statementVector_t parserBase::newKernelsFromLoops(statement &sKernel,
                                                      statementVector_t &omLoops,
                                                      varOriginMapVector_t &varDeps) {
      statementVector_t newKernels;

      const int kernelCount = (int) omLoops.size();

      varInfo &kernelVar         = *(sKernel.getFunctionVar());
      std::string kernelBaseName = kernelVar.name;

      globalScope->createUniqueSequentialVariables(kernelBaseName,
                                                   kernelCount);

      for (int k = 0; k < kernelCount; ++k) {
        // Copy newSKernel from sKernel
        statement &newSKernel = *(globalScope->makeSubStatement());

        newSKernel.info = sKernel.info;
        sKernel.expRoot.cloneTo(newSKernel.expRoot);

        newKernels.push_back(&newSKernel);

        statement &omLoop  = *(omLoops[k]);
        varOriginMap_t &deps = varDeps[k];

        varInfo &newKernelVar = *(new varInfo(kernelVar.clone()));
        newSKernel.setFunctionVar(newKernelVar);

        newKernelVar.name = kernelBaseName + occa::toString(k);

        addDepStatementsToKernel(newSKernel, deps);
        addDepsToKernelArguments(newSKernel, deps);
        addArgQualifiersTo(newSKernel);

        statement &sLaunch = launchStatementForKernel(sKernel,
                                                      omLoop,
                                                      k,
                                                      newKernelVar);

        newSKernel.addStatement(&sLaunch);

        // Swap positions for:
        //   outer-most-loop <--> host kernel
        statement::swapPlaces(omLoop, sLaunch);

        newSKernel.pushSourceLeftOf(omLoop.getStatementNode(),
                                    "occaParallelFor0");
      }

      return newKernels;
    }

    void parserBase::addDepStatementsToKernel(statement &sKernel,
                                              varOriginMap_t &deps) {

      varOriginMapIterator it = deps.begin();

      statementIdMap_t placedStatements;
      varInfoVector_t usedVars;

      while(it != deps.end()) {
        varInfo &var         = *(it->first);
        statement &varOrigin = *(it->second);

        if (var.hasQualifier("exclusive") ||
           var.hasQualifier("occaShared")) {

          usedVars.push_back(&var);
          placedStatements[&varOrigin];
        }

        ++it;
      }

      const int usedVarCount = (int) usedVars.size();

      for (int i = 0; i < usedVarCount; ++i)
        deps.erase(usedVars[i]);

      statementIdMapIterator sIt = placedStatements.begin();

      while(sIt != placedStatements.end()) {
        statement *firstS = ((sKernel.statementStart == NULL) ?
                             NULL :
                             sKernel.statementStart->value);

        statement *s2 = sIt->first->clone(&sKernel);

        sKernel.pushLeftOf(firstS, s2);

        ++sIt;
      }
    }

    void parserBase::addDepsToKernelArguments(statement &sKernel,
                                              varOriginMap_t &deps) {

      varInfo &kernelVar = *(sKernel.getFunctionVar());
      int argPos         = kernelVar.argumentCount;

      varToVarMap_t v2v;

      varOriginMapIterator it = deps.begin();

      while(it != deps.end()) {
        varInfo &var  = *(it->first);
        varInfo &var2 = *(var.clonePtr());

        v2v[&var] = &var2;
        kernelVar.addArgument(argPos++, var2);

        ++it;
      }

      sKernel.replaceVarInfos(v2v);
    }

    statement& parserBase::launchStatementForKernel(statement &sKernel,
                                                    statement &omLoop,
                                                    const int newKernelPos,
                                                    varInfo &newKernelVar) {
      statement &sHost = *(sKernel.makeSubStatement());
      sHost.info = smntType::blockStatement;

      std::stringstream ss;

      statementVector_t occaLoops = findOccaLoops(omLoop);
      const int occaLoopCount     = (int) occaLoops.size();

      attribute_t &maxOuterAttr = omLoop.attribute("occaMaxNest_outer");
      attribute_t &maxInnerAttr = omLoop.attribute("occaMaxNest_inner");

      const int outerDim = 1 + occa::atoi(maxOuterAttr.valueStr());
      const int innerDim = 1 + occa::atoi(maxInnerAttr.valueStr());

      // [outer/inner][dim]
      std::string iterExps[2][3];

      ss << "int outer, inner;\n";

      for (int i = 0; i < occaLoopCount; ++i) {
        statement &loop = *(occaLoops[i]);

        attribute_t &occaTagAttr  = loop.attribute("occaTag");
        attribute_t &occaNestAttr = loop.attribute("occaNest");

        const int occaTag  = (occaTagAttr.valueStr()  == "inner");
        const int occaNest = occa::atoi(occaNestAttr.valueStr());

        std::string &iterExp = iterExps[occaTag][occaNest];

        if (iterExp.size() == 0) {
          iterExp = loop.attribute("occaIterExp").valueStr();
        }
      }

      ss << "outer.dims = " << outerDim << ";\n";
      for (int o = 0; o < outerDim; ++o) {
        ss << "outer[" << o << "] = " << iterExps[0][o] << ";\n";
      }
      ss << "inner.dims = " << innerDim << ";\n";
      for (int i = 0; i < innerDim; ++i) {
        ss << "inner[" << i << "] = " << iterExps[1][i] << ";\n";
      }

      ss << "nestedKernels[" << newKernelPos << "].setRunDims(outer, inner);\n"
         << "nestedKernels[" << newKernelPos << "](";

      const int argCount = newKernelVar.argumentCount;

      // Skip occaKernelInfoArg
      for (int i = 1; i < argCount; ++i) {
        if (1 < i)
          ss << ", ";

        ss << newKernelVar.getArgument(i).name;
      }

      ss << ");";

      pushLanguage(parserInfo::parsingC);

      expNode allExp = splitAndPreprocessContent(ss.str());

      sHost.loadAllFromNode(allExp);

      // Change outer and inner types to occa::dim
      varInfo &outerVar = *(sHost.hasVariableInScope("outer"));
      varInfo &innerVar = *(sHost.hasVariableInScope("inner"));

      typeInfo &occaDimType = *(new typeInfo);
      occaDimType.name      = "occa::dim";

      outerVar.baseType = &occaDimType;
      innerVar.baseType = &occaDimType;

      popLanguage();

      return sHost;
    }

    void parserBase::storeKernelInfo(statement &sKernel,
                                     statementVector_t &newKernels) {

      const int kernelCount = (int) newKernels.size();

      varInfo &kernelVar    = *(sKernel.getFunctionVar());
      varInfo *newKernelVar = (kernelCount                     ?
                               newKernels[0]->getFunctionVar() :
                               NULL);

      const int argCount = kernelVar.argumentCount;

      kernelInfo &info = *(new kernelInfo);

      // Remove the 0 in the first new kernel
      //   to get the baseName
      info.name     = kernelVar.name;
      info.baseName = ((newKernelVar != NULL)                                      ?
                       newKernelVar->name.substr(0, newKernelVar->name.size() - 1) :
                       kernelVar.name);

      for (int k = 0; k < kernelCount; ++k) {
        info.nestedKernels.push_back(newKernels[k]);
      }
      for (int i = 0; i < argCount; ++i) {
        argumentInfo argInfo;
        varInfo &arg = kernelVar.getArgument(i);

        argInfo.pos     = i;
        argInfo.isConst = (arg.hasQualifier("occaConst") ||
                           arg.hasQualifier("occaConstant"));

        info.argumentInfos.push_back(argInfo);
      }

      kernelInfoMap[info.name] = &info;
    }

    void parserBase::zeroOccaIdsFrom(statement &s) {
      zeroOccaIdsFrom(s.expRoot);
    }

    void parserBase::zeroOccaIdsFrom(expNode &e) {
      expNode &flatRoot = *(e.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        if (((flatRoot[i].info & expType::presetValue) &&
            isAnOccaID(flatRoot[i].value))               ||
           (flatRoot[i].value.find("__occa_oTileVar") != std::string::npos)) {

          flatRoot[i].freeThis();

          flatRoot[i].info  = expType::presetValue;
          flatRoot[i].value = "0";
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void parserBase::addNestedKernelArgTo(statement &sKernel) {
      // Add nestedKernels argument
      varInfo &nestedKernelsArg = *(new varInfo());
      expNode nkNode = sKernel.createPlainExpNodeFrom("int *nestedKernels");
      nestedKernelsArg.loadFrom(nkNode);

      typeInfo &occaKernelType = *(new typeInfo);
      occaKernelType.name      = "occa::kernel";

      nestedKernelsArg.baseType = &occaKernelType;

      sKernel.addFunctionArg(1, nestedKernelsArg);
    }

    int parserBase::getKernelOuterDim(statement &s) {
      return getKernelDimFor(s, "outer");
    }

    int parserBase::getKernelInnerDim(statement &s) {
      return getKernelDimFor(s, "inner");
    }

    int parserBase::getKernelDimFor(statement &s,
                                    const std::string &tag) {
      statementNode *statementPos = s.statementStart;
      int dim = -1;

      attribute_t *occaTagAttr = s.hasAttribute("occaTag");

      const int passes = (1 + ((occaTagAttr != NULL) &&
                               occaTagAttr->valueStr() == tag));

      for (int pass = 0; pass < passes; ++pass) {
        attribute_t *occaNestAttr = ((pass == 0)                         ?
                                     s.hasAttribute("occaMaxNest_" + tag) :
                                     s.hasAttribute("occaNest"));

        if (occaTagAttr) {
          const std::string nestStr = occaNestAttr->valueStr();
          dim = ::atoi(nestStr.c_str());

          break;
        }
      }

      while(statementPos) {
        const int dim2 = getKernelDimFor(*(statementPos->value), tag);

        if (dim < dim2)
          dim = dim2;

        // Max dim
        if (dim == 2)
          return dim;

        statementPos = statementPos->right;
      }

      return dim;
    }

    int parserBase::getOuterMostForDim(statement &s) {
      return getForDim(s, "outer");
    }

    int parserBase::getInnerMostForDim(statement &s) {
      return getForDim(s, "inner");
    }

    int parserBase::getForDim(statement &s,
                              const std::string &tag) {

      attribute_t *occaNestAttr = s.hasAttribute("occaMaxNest_" + tag);

      OCCA_ERROR("Error, outer-most loop doesn't contain [" << tag << "] loops",
                 (occaNestAttr != NULL) &&
                 (occaNestAttr->valueStr() == tag));

      const std::string tagStr = occaNestAttr->valueStr();
      return ::atoi(tagStr.c_str());
    }

    void parserBase::splitDefineForVariable(varInfo &var) {
      statement &origin = *(varOriginMap[&var]);

      // Ignore kernel arguments
      if (origin.info & smntType::functionStatement)
        return;

      int argc   = origin.getDeclarationVarCount();
      int argPos = 0;

      for (int i = 0; i < argc; ++i) {
        varInfo &argVar = origin.getDeclarationVarInfo(i);

        if (&argVar == &var) {
          argPos = i;
          break;
        }
      }

      if (argPos) {
        statement &s        = origin.pushNewStatementLeft(smntType::declareStatement);
        s.expRoot.info      = origin.expRoot.info;
        s.expRoot.leaves    = new expNode*[argPos];
        s.expRoot.leafCount = argPos;

        for (int i = 0; i < argPos; ++i) {
          varInfo &argVar = origin.getDeclarationVarInfo(i);

          s.expRoot.leaves[i]   = origin.expRoot.leaves[i];
          varOriginMap[&argVar] = &s;
        }
      }

      if ((argPos + 1) < argc) {
        const int newLeafCount = (argc - (argPos + 1));

        statement &s        = origin.pushNewStatementRight(smntType::declareStatement);
        s.expRoot.info      = origin.expRoot.info;
        s.expRoot.leaves    = new expNode*[newLeafCount];
        s.expRoot.leafCount = newLeafCount;

        for (int i = 0; i < newLeafCount; ++i) {
          varInfo &argVar = origin.getDeclarationVarInfo(argPos + 1 + i);

          s.expRoot.leaves[i]   = origin.expRoot.leaves[argPos + 1 + i];
          varOriginMap[&argVar] = &s;

          // Print out type for the new statement
          if (i == 0)
            s.expRoot.getVariableInfoNode(0)->info |= expType::type;
        }
      }

      origin.expRoot.leaves[0] = origin.expRoot.leaves[argPos];
      origin.expRoot.leafCount = 1;

      // Print out type for the new statement
      origin.expRoot.getVariableInfoNode(0)->info |= expType::type;
    }

    void parserBase::splitDefineAndInitForVariable(varInfo &var) {
      statement &origin = *(varOriginMap[&var]);

      // Ignore kernel arguments
      if (origin.info & smntType::functionStatement)
        return;

      int argc = origin.getDeclarationVarCount();

      // Make sure var is the only variable
      if (1 < argc)
        splitDefineForVariable(var);

      // Return if [var] is not being initialized
      if (!origin.expRoot.variableHasInit(0))
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
      // expNode *tmp = &(origin.expRoot[0]); [--]

      expNode::swap(origin.expRoot, origin.expRoot[0]);
      expNode::swap(origin.expRoot, s.expRoot);

      // delete tmp; [--] ?
      //================================

      if (s.expRoot.lastLeaf()->value != ";")
        s.expRoot.addNode(expType::endStatement, ";");

      varOriginMap[&var] = &s;
    }

    void parserBase::addInnerFors(statement &s) {
      int innerDim = getKernelInnerDim(s);

      if (innerDim < 0)
        return;

      varInfoIdMap_t varInfoIdMap;
      int currentInnerID = 0;

      // Add inner for-loops
      addInnerForsTo(s, varInfoIdMap, currentInnerID, innerDim);
    }

    void parserBase::addInnerForsTo(statement &s,
                                    varInfoIdMap_t &varInfoIdMap,
                                    int &currentInnerID,
                                    const int innerDim) {

      statementNode *ssStart = s.statementStart;
      statementNode *ssEnd   = lastNode(ssStart);

      statementNode *statementPos = ssStart;

      std::vector<statement*> sBreaks;
      std::vector<int> cInnerIDs;

      cInnerIDs.push_back(currentInnerID);

      while(statementPos) {
        statement &s2 = *(statementPos->value);

        // Add inner-for inside the for/while loop
        if (s2.info & (smntType::forStatement |
                      smntType::whileStatement)) {

          addInnerForsTo(s2, varInfoIdMap, currentInnerID, innerDim);
          sBreaks.push_back(&s2);
          cInnerIDs.push_back(++currentInnerID);
        }
        else if (s2.hasBarrier()) {
          sBreaks.push_back(&s2);
          cInnerIDs.push_back(++currentInnerID);
        }

        statementPos = statementPos->right;
      }

      sBreaks.push_back(NULL);

      const int oldBreaks = sBreaks.size();
      int breaks = oldBreaks;

      // Start with first non-break statement
      for (int b = 0; b < breaks; ++b) {
        if (ssStart &&
           (sBreaks[b] == ssStart->value)) {

          ssStart = ssStart->right;

          --b;
          --breaks;
        }
        else
          break;
      }

      if (breaks == 0)
        return;

      // Remove first non-break statement info
      const int deltaBreaks = (oldBreaks - breaks);

      for (int b = 0; b < breaks; ++b) {
        sBreaks[b]   = sBreaks[deltaBreaks + b];
        cInnerIDs[b] = cInnerIDs[deltaBreaks + b];
      }

      for (int b = 0; b < breaks; ++b) {
        ssEnd = ssStart;

        if (ssEnd == NULL)
          break;

        const int cInnerID = cInnerIDs[b];

        // Find statements for inner-loop
        while(ssEnd &&
              ssEnd->value != sBreaks[b]) {

          ssEnd = ssEnd->right;
        }

        statement *outerInnerS = NULL;
        statement *innerInnerS = NULL;

        // Add inner-for loops
        for (int i = 0; i <= innerDim; ++i) {
          const int innerID = (innerDim - i);

          statement *newInnerS = new statement(smntType::occaFor,
                                               (outerInnerS ? outerInnerS : &s));

          if (outerInnerS == NULL) {
            outerInnerS = newInnerS;
            innerInnerS = newInnerS;
          }

          newInnerS->expRoot.info   = expType::occaFor;
          newInnerS->expRoot.value  = "occaInnerFor";
          newInnerS->expRoot.value += ('0' + innerID);

          if (innerInnerS != newInnerS) {
            innerInnerS->addStatement(newInnerS);
            innerInnerS = newInnerS;
          }
        }

        statementNode *sn = ssStart;

        while(ssStart != ssEnd) {
          statement &s2 = *(ssStart->value);

          innerInnerS->addStatement(&s2);

          checkStatementForExclusives(s2,
                                      varInfoIdMap,
                                      cInnerID);

          // Save this SN for outerInnerS
          if (ssStart != sn)
            popAndGoRight(ssStart);
          else
            ssStart = ssStart->right;
        }

        // Move to the right of the break
        if (ssStart)
          ssStart = ssStart->right;

        sn->value = outerInnerS;

        // Skip breaks
        while(ssStart            &&
              ((b + 1) < breaks) &&
              ssStart->value == sBreaks[b + 1]) {

          ssStart = ssStart->right;
          ++b;
        }
      } // for (breaks)
    }

    void parserBase::checkStatementForExclusives(statement &s,
                                                 varInfoIdMap_t &varInfoIdMap,
                                                 const int innerID) {

      statementNode *statementPos = s.statementStart;

      while(statementPos) {
        checkStatementForExclusives(*(statementPos->value),
                                    varInfoIdMap,
                                    innerID);

        statementPos = statementPos->right;
      }

      // Find variables
      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        varInfo *sVar = NULL;

        // Check for variable
        if ((flatRoot[i].info & expType::varInfo) &&
           (flatRoot[i].info & expType::declaration)) {

          sVar = &(flatRoot[i].getVarInfo());
        }

        // Has variable
        if ((sVar != NULL)                   &&
           !sVar->hasQualifier("occaShared") &&
           !sVar->hasQualifier("exclusive")) {

          varInfoIdMapIterator it = varInfoIdMap.find(sVar);

          if (it != varInfoIdMap.end()) {
            if (((it->second) != -1) &&
               ((it->second) != innerID)) {

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

    void parserBase::addOuterFors(statement &s) {
      int outerDim = getKernelOuterDim(s);

      if (outerDim < 0)
        return;

      statement *sPos = &s;

      for (int o = outerDim; 0 <= o; --o) {
        statement *newStatement = new statement(smntType::occaFor, &s);

        newStatement->expRoot.info   = expType::printValue;
        newStatement->expRoot.value  = "occaOuterFor";
        newStatement->expRoot.value += ('0' + o);

        newStatement->scope = sPos->scope;

        statementNode *sn = sPos->statementStart;

        while(sn) {
          newStatement->addStatement(sn->value);

          sn->value->up = newStatement;

          statementNode *sn2 = sn->right;

          delete sn;
          sn = sn2;
        }

        sPos->statementStart = sPos->statementEnd = NULL;
        sPos->scope = new scopeInfo();

        sPos->addStatement(newStatement);

        sPos = newStatement;
      }
    }

    void parserBase::removeUnnecessaryBlocksInKernel(statement &s) {
      statement *sPos = &s;

      // Get rid of empty blocks
      //  kernel void blah() {{  -->  kernel void blah() {
      //  }}                    -->  }
      while(sPos->statementCount() == 1) {
        statement *sDown = sPos->statementStart->value;

        if (sDown->info == smntType::blockStatement) {
          sPos->scope->appendVariablesFrom(sDown->scope);

          sPos->statementStart = sPos->statementEnd = NULL;

          statementNode *sn = sDown->statementStart;

          while(sn) {
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

    void parserBase::addOccaForsToKernel(statement &s) {
      addInnerFors(s);
      addOuterFors(s);
    }

    void parserBase::addOccaFors() {
      statementNode *statementPos = globalScope->statementStart;

      while(statementPos) {
        statement *s = statementPos->value;

        if (statementIsAKernel(*s)            && // Kernel
           (s->statementStart != NULL)       && //   not empty
           !statementKernelUsesNativeOKL(*s) && //   not OKL
           !statementKernelUsesNativeOCCA(*s)) { //   not OCCA

          removeUnnecessaryBlocksInKernel(*s);
          addOccaForsToKernel(*s);
        }

        statementPos = statementPos->right;
      }
    }

    void parserBase::setupOccaVariables(statement &s) {
      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
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

        if (isId || isDim) {
          std::string loopTag, loopNest;

          if (isId) {
            // ioLoop is not capitalized to be consistent with
            //   attribute names

            if (isInnerId || isOuterId) {
              // [occa][-----][Id#]
              loopTag     = value.substr(4,5);
              loopTag[0] -= ('A' - 'a');

              // [occa][-----Id][#]
              loopNest = value.substr(11,1);

              s.updateOccaOMLoopAttributes(loopTag, loopNest);
            }
            else { // isGlobalId
              // [occa][------Id][#]
              loopNest = value.substr(12,1);

              s.updateOccaOMLoopAttributes("inner", loopNest);
              s.updateOccaOMLoopAttributes("outer", loopNest);
            }
          }
          else { // isDim
            if (isInnerDim || isOuterDim) {
              // [occa][-----][Dim#]
              loopTag     = value.substr(4,5);
              loopTag[0] -= ('A' - 'a');

              // [occa][-----Dim][#]
              loopNest = value.substr(12,1);

              s.updateOccaOMLoopAttributes(loopTag, loopNest);
            }
            else { // isGlobalDim
              // [occa][------Dim][#]
              loopNest = value.substr(13,1);

              s.updateOccaOMLoopAttributes("inner", loopNest);
              s.updateOccaOMLoopAttributes("outer", loopNest);
            }
          }

        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    //---[ Operator Information ]-----------------
    varInfo* parserBase::hasOperator(const info_t expInfo,
                                     const std::string &op,
                                     varInfo &var) {
      return NULL;
    }

    varInfo* parserBase::hasOperator(const info_t expInfo,
                                     const std::string &op,
                                     varInfo &varL,
                                     varInfo &varR) {
      return NULL;
    }

    varInfo parserBase::thVarInfo(const info_t thType) {
      varInfo var;

      const std::string str = typeHolder::typeToBaseTypeStr(thType);

      var.baseType = globalScope->hasTypeInScope(str);

      return var;
    }

    varInfo parserBase::thOperatorReturnType(const info_t expInfo,
                                             const std::string &op,
                                             const info_t thType) {
      typeHolder th, ret;
      varInfo var;

      th.type = thType;

      if (expInfo & expType::L)
        ret = applyLOperator(op, th);
      else
        ret = applyROperator(th, op);

      var.baseType = globalScope->hasTypeInScope(ret.baseTypeStr());

      if (ret.isUnsigned())
        var.addQualifier("unsigned");

      if (ret.isALongInt())
        var.addQualifier("long");

      return var;
    }

    varInfo parserBase::thOperatorReturnType(const info_t expInfo,
                                             const std::string &op,
                                             const info_t thTypeL,
                                             const info_t thTypeR) {
      varInfo var;

      if (op == "(") {
        OCCA_ERROR("Cannot use () operator with void* (Example: 10(10))",
                   false);
      }
      else if (op == "[") {
        OCCA_ERROR("Cannot use [] operator with void* (Example: 10[10])",
                   false);
      }
      else {
        typeHolder l, r;

        l.type = thTypeL;
        r.type = thTypeR;

        typeHolder ret = applyLROperator(l, op, r);

        var.baseType = globalScope->hasTypeInScope(ret.baseTypeStr());

        if (ret.isUnsigned())
          var.addQualifier("unsigned");

        if (ret.isALongInt())
          var.addQualifier("long");
      }

      return var;
    }
    //============================================

    bool isAnOccaID(const std::string &s) {
      return (isAnOccaInnerID(s) ||
              isAnOccaOuterID(s) ||
              isAnOccaGlobalID(s));
    }

    bool isAnOccaDim(const std::string &s) {
      return (isAnOccaInnerDim(s) ||
              isAnOccaOuterDim(s) ||
              isAnOccaGlobalDim(s));
    }

    bool isAnOccaInnerID(const std::string &s) {
      return ((s == "occaInnerId0") ||
              (s == "occaInnerId1") ||
              (s == "occaInnerId2"));
    }

    bool isAnOccaOuterID(const std::string &s) {
      return ((s == "occaOuterId0") ||
              (s == "occaOuterId1") ||
              (s == "occaOuterId2"));
    }

    bool isAnOccaGlobalID(const std::string &s) {
      return ((s == "occaGlobalId0") ||
              (s == "occaGlobalId1") ||
              (s == "occaGlobalId2"));
    }

    bool isAnOccaInnerDim(const std::string &s) {
      return ((s == "occaInnerDim0") ||
              (s == "occaInnerDim1") ||
              (s == "occaInnerDim2"));
    }

    bool isAnOccaOuterDim(const std::string &s) {
      return ((s == "occaOuterDim0") ||
              (s == "occaOuterDim1") ||
              (s == "occaOuterDim2"));
    }

    bool isAnOccaGlobalDim(const std::string &s) {
      return ((s == "occaGlobalDim0") ||
              (s == "occaGlobalDim1") ||
              (s == "occaGlobalDim2"));
    }

    //==============================================

    expNode splitContent(const std::string &str, const int parsingLanguage_) {
      return splitContent(str.c_str(), parsingLanguage_);
    }

    expNode splitContent(const char *cRoot, const int parsingLanguage_) {
      pushLanguage(parsingLanguage_);

      const bool parsingFortran   = (parsingLanguage_ & parserInfo::parsingFortran);
      const char continuationChar = (parsingFortran ? '&' : '\\');

      const char *c = cRoot;

      int lineCount = 1 + countDelimiters(c, '\n');
      info_t status = readingCode;

      expNode allExp;
      allExp.addNodes(lineCount);
      int expPos = 0;

      std::string line;

      while(*c != '\0') {
        const char *cEnd = readLine(c, parsingLanguage_);

        line += compressAllWhitespace(c, cEnd - c, parsingLanguage_);
        c = cEnd;

        // Line carrying over to next line
        if (line.size()                                 &&
           (line[line.size() - 1] == continuationChar) &&
           (*cEnd != '\0')) {

          line.erase(line.size() - 1);
          continue;
        }

        if (line.size()) {
          // Remove old-style Fortran line comment
          if (parsingFortran &&
             (line[0] == 'c')) {

            line.clear();
            continue;
          }

          status = stripComments(line, parsingLanguage_);
          compressAllWhitespace(line, parsingLanguage_);

          if (line.size() &&
             ((status != insideCommentBlock) ||
              (status == finishedCommentBlock))) {

            allExp[expPos++].value = line;
            line.clear();
          }
        }
      }

      allExp.leafCount = expPos;

      popLanguage();

      return allExp;
    }

    expNode splitAndLabelContent(const std::string &str, const int parsingLanguage_) {
      return splitAndLabelContent(str.c_str(), parsingLanguage_);
    }

    expNode splitAndLabelContent(const char *cRoot, const int parsingLanguage_) {
      expNode allExp = splitContent(cRoot, parsingLanguage_);

      return labelCode(allExp);
    }

    expNode splitAndOrganizeContent(const std::string &str, const int parsingLanguage_) {
      return splitAndOrganizeContent(str.c_str(), parsingLanguage_);
    }

    expNode splitAndOrganizeContent(const char *cRoot, const int parsingLanguage_) {
      expNode tmpExp, retExp;

      tmpExp = splitContent(cRoot, parsingLanguage_);
      tmpExp = labelCode(tmpExp, parsingLanguage_);

      retExp.loadFromNode(tmpExp);

      return retExp;
    }

    expNode& labelCode(expNode &allExp, const int parsingLanguage_) {
      pushLanguage(parsingLanguage_);

      const bool parsingC       = (parsingLanguage_ & parserInfo::parsingC      );
      const bool parsingFortran = (parsingLanguage_ & parserInfo::parsingFortran);

      const bool usingLeaves = (allExp.leafCount != 0);
      const int lineCount    = (usingLeaves      ?
                                allExp.leafCount :
                                1);

      const bool addSpace = true; // For readability

      expNode node, *cNode = &node;

      for (int linePos = 0; linePos < lineCount; ++linePos) {
        expNode &lineNode = (usingLeaves ? allExp[linePos] : allExp);

        const std::string &line = lineNode.value;
        const char *cLeft = line.c_str();

        while(*cLeft != '\0') {
          skipWhitespace(cLeft);

          const char *cRight = cLeft;

          bool loadString = isAString(cLeft);
          bool loadNumber = ((*cLeft != '-') &&
                             (*cLeft != '+') &&
                             isANumber(cLeft));

          if (loadString) { //-----------------------------------------------[ 1 ]
            skipString(cRight, parsingLanguage_);

            cNode->addNode(expType::presetValue);
            cNode->lastNode().value = std::string(cLeft, (cRight - cLeft));

            cLeft = cRight;
          }
          else if (loadNumber) { //------------------------------------------[ 2 ]
            typeHolder th;
            th.load(cLeft);

            // skipNumber(cRight, parsingLanguage); [---]

            cNode->addNode(expType::presetValue);

            cNode->lastNode().value = (std::string) th;
            cNode->lastNode().info  = expType::firstPass | expType::presetValue;
          }
          else { //---------------------------------------------------------[ 3 ]
            const int delimiterChars = isAWordDelimiter(cLeft, parsingLanguage_);

            cNode->addNode();

            expNode &lastExpNode     = (*cNode)[-1];
            std::string &lastNodeStr = lastExpNode.value;

            if (delimiterChars) { //---------------------------------------[ 3.1 ]
              if (parsingFortran) { //-------------------------------------[ 3.1.1 ]
                // Translate Fortran keywords
                std::string op(cLeft, delimiterChars);
                std::string upOp = upString(op);

                if (upOp[0] == '.') {
                  if     (upOp == ".TRUE.")  upOp = "true";
                  else if (upOp == ".FALSE.") upOp = "false";

                  else if (upOp == ".LT.")    upOp = "<";
                  else if (upOp == ".GT.")    upOp = ">";
                  else if (upOp == ".LE.")    upOp = "<=";
                  else if (upOp == ".GE.")    upOp = ">=";
                  else if (upOp == ".EQ.")    upOp = "==";

                  else if (upOp == ".NOT.")   upOp = "!";
                  else if (upOp == ".AND.")   upOp = "&&";
                  else if (upOp == ".OR.")    upOp = "||";
                  else if (upOp == ".EQV.")   upOp = "==";
                  else if (upOp == ".NEQV.")  upOp = "!=";

                  lastNodeStr = upOp;
                }
                else if (upOp == "/=") {
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

              if (lastExpNode.info & expType::C) { //----------------------[ 3.1.3 ]
                if (charStartsSection(lastExpNode.value[0])) {
                  cNode = &(cNode->lastNode());
                }
                else {
                  cNode->removeNode(-1);
                  cNode = cNode->up;
                }
              } //=======================================================[ 3.1.3 ]
              else if (lastExpNode.info & expType::macroKeyword) { //------[ 3.1.4 ]
                lastNodeStr = line;

                cLeft = line.c_str() + strlen(line.c_str()) - delimiterChars;
              } //=======================================================[ 3.1.4 ]

              cLeft += delimiterChars;
            } //=========================================================[ 3.1 ]
            else { //-----------------------------------------------------[ 3.2 ]
              skipWord(cRight, parsingLanguage_);

              std::string str(cLeft, (cRight - cLeft));
              keywordTypeMapIterator it;

              if (parsingFortran) {
                std::string upStr = upString(str);

                it = keywordType->find(upStr);

                if (it != keywordType->end())
                  str = upStr;
              }
              else {
                it = keywordType->find(str);
              }

              lastNodeStr = str;

              if (it == keywordType->end())
                lastExpNode.info = expType::firstPass | expType::unknown;
              else {
                lastExpNode.info = it->second;

                if (parsingC) {
                  if (checkLastTwoNodes(*cNode, "else", "if")) {
                    mergeLastTwoNodes(*cNode);
                  }
                }
                else {
                  if (checkLastTwoNodes(*cNode, "else", "if"   , parsingLanguage_) ||
                     checkLastTwoNodes(*cNode, "do"  , "while", parsingLanguage_)) {

                    mergeLastTwoNodes(*cNode, addSpace, parsingLanguage_);
                  }
                  else if (checkLastTwoNodes(*cNode, "end" , "do"        , parsingLanguage_) ||
                          checkLastTwoNodes(*cNode, "end" , "if"        , parsingLanguage_) ||
                          checkLastTwoNodes(*cNode, "end" , "function"  , parsingLanguage_) ||
                          checkLastTwoNodes(*cNode, "end" , "subroutine", parsingLanguage_)) {

                    mergeLastTwoNodes(*cNode, !addSpace, parsingLanguage_);

                    expNode &mergedNode = (*cNode)[-1];

                    mergedNode.info = (*keywordType)[mergedNode.value];
                  }
                }
              }

              cLeft = cRight;
            } //=========================================================[ 3.2 ]
          } //===========================================================[ 3 ]
        }

        if (parsingFortran) {
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
                           const int parsingLanguage_) {

      if (parsingLanguage_ & parserInfo::parsingC) {
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
                           const int parsingLanguage_) {

      if (node.leafCount < 2)
        return;

      if (addSpace)
        node[-2].value += ' ';

      node[-2].value += node[-1].value;

      if (parsingLanguage_ & parserInfo::parsingFortran)
        node[-2].value = upString(node[-2].value);

      node.removeNode(-1);
    }

    expNode createExpNodeFrom(const std::string &source) {
      return statement::createExpNodeFrom(source);
    }

    expNode createOrganizedExpNodeFrom(const std::string &source) {
      expNode ret = statement::createExpNodeFrom(source);
      ret.organize();

      return ret;
    }

    void loadKeywords(const int parsingLanguage_) {
      if (parsingLanguage_ & parserInfo::parsingC)
        loadCKeywords();
      else
        loadFortranKeywords();
    }

    void loadCKeywords() {
      initCKeywords();

      keywordType  = &cKeywordType;
      opPrecedence = &cOpPrecedence;

      for (int i = 0; i < maxOpLevels; ++i) {
        opLevelMap[i] = &cOpLevelMap[i];
        opLevelL2R[i] = &cOpLevelL2R[i];
      }
    }

    void loadFortranKeywords() {
      initFortranKeywords();

      keywordType  = &fortranKeywordType;
      opPrecedence = &fortranOpPrecedence;

      for (int i = 0; i < maxOpLevels; ++i) {
        opLevelMap[i] = &fortranOpLevelMap[i];
        opLevelL2R[i] = &fortranOpLevelL2R[i];
      }
    }

    void initCKeywords() {
      if (cKeywordsAreInitialized)
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
      cKeywordType[":"]                  = expType::LR;
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

      cKeywordType["long"]               = expType::qualifier;
      cKeywordType["short"]              = expType::qualifier;
      cKeywordType["signed"]             = expType::qualifier;
      cKeywordType["unsigned"]           = expType::qualifier;

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
      cKeywordType["@"]                  = expType::L;
      cKeywordType["__attribute__"]      = expType::L;
      cKeywordType["__asm"]              = expType::specialKeyword;

      //---[ C++ ]----------------------
      cKeywordType["virtual"]            = expType::qualifier;

      cKeywordType["namespace"]          = expType::namespace_;

      //---[ Constants ]------------------
      cKeywordType["..."]                = expType::presetValue;
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
      cKeywordType["throw"]              = expType::specialKeyword;

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

      std::string mathFunctions[18] = {
        "min" , "max"  ,
        "sqrt", "sin"  , "asin" ,
        "sinh", "asinh", "cos"  ,
        "acos", "cosh" , "acosh",
        "tan" , "atan" , "tanh" ,
        "atanh", "exp" , "log2" ,
        "log10"
      };

      for (int i = 0; i < 18; ++i) {
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

      while(it != cKeywordType.end()) {
        it->second |= expType::firstPass;
        ++it;
      }
    }

    void initFortranKeywords() {
      if (fortranKeywordsAreInitialized)
        return;

      fortranKeywordsAreInitialized = true;

      //---[ Operator Info ]--------------
      fortranKeywordType["@"]  = expType::L;

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

      for (int i = 0; i < 5; ++i) {
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

      for (int i = 0; i < 16; ++i)
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

      while(it != fortranKeywordType.end()) {
        it->second |= expType::firstPass;
        ++it;
      }
    }

    //---[ OCCA Loop Info ]-------------
    occaLoopInfo::occaLoopInfo(statement &s,
                               const int parsingLanguage_,
                               const std::string &tag) {
      parsingLanguage = parsingLanguage_;

      lookForLoopFrom(s, tag);
    }

    void occaLoopInfo::lookForLoopFrom(statement &s,
                                       const std::string &tag) {
      sInfo = &s;

      while(sInfo) {
        if (sInfo->info == smntType::occaFor) {
          attribute_t *occaTagAttr = sInfo->hasAttribute("occaTag");

          if (occaTagAttr != NULL)
            break;
        }

        sInfo = sInfo->up;
      }

      if (s.parser.parsingLanguage & parserInfo::parsingFortran)
        return;

      //---[ Overload iter vars ]---
      setIterDefaultValues();

      expNode &node1   = *(sInfo->getForStatement(0));
      expNode &node2   = *(sInfo->getForStatement(1));
      expNode &node3   = *(sInfo->getForStatement(2));

      //---[ Node 1 Check ]---
      OCCA_ERROR("Wrong 1st statement for:\n  " << sInfo->expRoot,
                 (node1.info == expType::declaration) &&
                 (node1.getVariableCount() == 1)      &&
                 node1.variableHasInit(0));

      varInfo &iterVar = node1.getVariableInfoNode(0)->getVarInfo();

      std::string &iter = iterVar.name;

      if ( !iterVar.hasQualifier("occaConst") )
        iterVar.addQualifier("occaConst");

      //---[ Node 2 Check ]---
      OCCA_ERROR("Wrong 2nd statement for:\n  " << sInfo->expRoot,
                 (node2.leafCount == 1) &&
                 ((node2[0].value == "<=") ||
                  (node2[0].value == "<" ) ||
                  (node2[0].value == ">" ) ||
                  (node2[0].value == ">=")));

      if (parsingLanguage & parserInfo::parsingC) {
        OCCA_ERROR("Wrong 2nd statement for:\n  " << sInfo->expRoot,
                   (node2[0][0].value == iter) ||
                   (node2[0][1].value == iter));
      }

      //---[ Node 3 Check ]---
      OCCA_ERROR("Wrong 3rd statement for:\n  " << sInfo->expRoot,
                 (node3.leafCount == 1) &&
                 ((node3[0].value == "++") ||
                  (node3[0].value == "--") ||
                  (node3[0].value == "+=") ||
                  (node3[0].value == "-=")));

      OCCA_ERROR("Wrong 3rd statement for:\n  " << sInfo->expRoot,
                 (node3[0][0].value == iter) ||
                 (node3[0][1].value == iter));
    }

    // [-] Missing
    void occaLoopInfo::loadForLoopInfo(int &innerDims, int &outerDims,
                                       std::string *innerIters,
                                       std::string *outerIters) {
    }

    void occaLoopInfo::getLoopInfo(std::string &loopTag,
                                   std::string &loopNest) {

      attribute_t &occaTagAttr  = sInfo->attribute("occaTag");
      attribute_t &occaNestAttr = sInfo->attribute("occaNest");

      // [-----][#]
      loopTag  = occaTagAttr.valueStr();
      loopNest = occaNestAttr.valueStr();
    }

    void occaLoopInfo::getLoopNode1Info(std::string &iter,
                                        std::string &start) {

      expNode &node1 = *(sInfo->getForStatement(0));

      if (parsingLanguage & parserInfo::parsingC) {
        varInfo &iterVar = node1.getVariableInfoNode(0)->getVarInfo();

        iter  = iterVar.name;
        start = *(node1.getVariableInitNode(0));
      }
      else {
        iter  = node1[0][0].getVarInfo().name;
        start = node1[0][1].getVarInfo().name;
      }
    }

    void occaLoopInfo::getLoopNode2Info(std::string &bound,
                                        std::string &iterCheck) {

      expNode &node2 = *(sInfo->getForStatement(1));

      iterCheck = node2[0].value;

      if (parsingLanguage & parserInfo::parsingC) {
        if ((iterCheck == "<=") || (iterCheck == "<"))
          bound = (std::string) node2[0][1];
        else
          bound = (std::string) node2[0][0];
      }
      else {
        std::string iter, start;
        getLoopNode1Info(iter, start);

        // [doStart][#]
        std::string suffix = start.substr(7);

        bound = "(doEnd" + suffix + " + 1)";
      }
    }

    void occaLoopInfo::getLoopNode3Info(std::string &stride,
                                        std::string &strideOpSign,
                                        std::string &strideOp) {
      expNode &node3 = *(sInfo->getForStatement(2));

      std::string iter, tmp;
      getLoopNode1Info(iter, tmp);

      strideOp = node3[0].value;

      // [+]+, [+]=
      // [-]-, [-]=
      strideOpSign = strideOp[0];

      if ((strideOp == "++") || (strideOp == "--")) {
        stride = "1";
      }
      else {
        if (node3[0][0].getVarInfo().name == iter)
          stride = (std::string) node3[0][1];
        else
          stride = (std::string) node3[0][0];
      }
    }

    void occaLoopInfo::setIterDefaultValues() {
      int innerDims, outerDims;
      std::string innerIters[3], outerIters[3];

      loadForLoopInfo(innerDims, outerDims,
                      innerIters, outerIters);
    }

    std::string occaLoopInfo::getSetupExpression() {
      expNode &node1 = *(sInfo->getForStatement(0));

      return node1.toString("", (expFlag::noSemicolon |
                                 expFlag::noNewline));
    }
    //==================================
  }
}
