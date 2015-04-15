#ifndef OCCA_PARSER_HEADER
#define OCCA_PARSER_HEADER

#include "occaParserDefines.hpp"
#include "occaParserMacro.hpp"
#include "occaParserTools.hpp"
#include "occaParserNodes.hpp"
#include "occaParserTypes.hpp"
#include "occaParserStatement.hpp"
#include "occaParserMagic.hpp"
#include "occaTools.hpp"

namespace occa {
  namespace parserNS {
    class occaLoopInfo;

    class parserBase {
    public:
      std::string filename;

      bool parsingC;

      macroMap_t macroMap;
      std::vector<macroInfo> macros;

      bool macrosAreInitialized;

      varUsedMap_t varUpdateMap;
      varUsedMap_t varUsedMap;     // Statements are placed backwards

      kernelInfoMap_t kernelInfoMap;

      statement *globalScope;

      //---[ Warnings ]-----------------
      bool warnForMissingBarriers;
      bool warnForBarrierConditionals;
      bool magicEnabled;
      //================================

      parserBase();
      inline ~parserBase(){}

      const std::string parseFile(const std::string &filename,
                                  const bool parsingC_ = true);

      const std::string parseSource(const char *cRoot);

      //---[ Macro Parser Functions ]---
      std::string getMacroName(const char *&c);
      std::string getMacroIncludeFile(const char *&c);

      typeHolder evaluateMacroStatement(const char *&c);
      bool evaluateMacroBoolStatement(const char *&c);
      static typeHolder evaluateLabelNode(strNode *labelNodeRoot);

      void loadMacroInfo(macroInfo &info, const char *&c);

      int loadMacro(strNode *nodePos, const int state = doNothing);
      int loadMacro(const std::string &line, const int state = doNothing);
      int loadMacro(strNode *nodePos, const std::string &line, const int state = doNothing);

      void applyMacros(std::string &line);

      strNode* preprocessMacros(strNode *nodeRoot);

      strNode* splitAndPreprocessContent(const std::string &s);
      strNode* splitAndPreprocessContent(const char *cRoot);
      strNode* splitAndPreprocessFortranContent(const char *cRoot);
      //====================================

      void initMacros(const bool parsingC = true);
      void initFortranMacros();

      void loadLanguageTypes();

      void applyToAllStatements(statement &s,
                                applyToAllStatements_t func);

      void applyToAllKernels(statement &s,
                             applyToAllStatements_t func);

      void applyToStatementsDefiningVar(applyToStatementsDefiningVar_t func);

      void applyToStatementsUsingVar(varInfo &info,
                                     applyToStatementsUsingVar_t func);

      bool statementIsAKernel(statement &s);

      statement* getStatementKernel(statement &s);
      statement* getStatementOuterMostLoop(statement &s);

      bool statementKernelUsesNativeOCCA(statement &s);

      bool statementKernelUsesNativeOKL(statement &s);

      bool statementKernelUsesNativeLanguage(statement &s);

      void addOccaForCounter(statement &s,
                             const std::string &ioLoop,
                             const std::string &loopNest,
                             const std::string &loopIters = "");

      void setupOccaFors(statement &s);

      bool statementIsOccaOuterFor(statement &s);
      bool statementIsOccaInnerFor(statement &s);

      bool statementHasOccaOuterFor(statement &s);
      bool statementHasOccaFor(statement &s);

      bool statementHasOklFor(statement &s);

      bool statementHasOccaStuff(statement &s);

      void splitTileOccaFors(statement &s);

      void markKernelFunctions();
      void labelNativeKernels();

      void setupCudaVariables(statement &s);

      void addFunctionPrototypes();

      int statementOccaForNest(statement &s);
      bool statementIsAnOccaFor(statement &s);

      void checkOccaBarriers(statement &s);
      void addOccaBarriers();
      void addOccaBarriersToStatement(statement &s);

      bool statementHasBarrier(statement &s);

      void addParallelFors(statement &s);

      void updateConstToConstant();

      strNode* occaExclusiveStrNode(varInfo &info,
                                    const int depth,
                                    const int sideDepth);

      void addArgQualifiers();

      void floatSharedAndExclusivesUp(statement &s);
      statementNode* appendSharedAndExclusives(statement &s,
                                               statementNode *snTail,
                                               bool isAppending = false);

      void modifyExclusiveVariables(statement &s);

      void modifyTextureVariables();

      statementNode* splitKernelStatement(statementNode *snKernel,
                                          kernelInfo &info);

      statementNode* getOuterLoopsInStatement(statement &s);
      statementNode* getOccaLoopsInStatement(statement &s,
                                             const bool getNestedLoops = true);

      int kernelCountInOccaLoops(statementNode *occaLoops);

      void zeroOccaIdsFrom(statement &s);

      statementNode* createNestedKernelsFromLoops(statementNode *snKernel,
                                                  kernelInfo &info,
                                                  statementNode *outerLoopRoot);

      std::string getNestedKernelArgsFromLoops(statement &sKernel);

      void setupHostKernelArgsFromLoops(statement &sKernel);

      void loadKernelInfos();

      void stripOccaFromKernel(statement &s);

      std::string occaScope(statement &s);

      void incrementDepth(statement &s);

      void decrementDepth(statement &s);

      statementNode* findStatementWith(statement &s,
                                       findStatementWith_t func);

      static int getKernelOuterDim(statement &s);
      static int getKernelInnerDim(statement &s);

      int getOuterMostForDim(statement &s);
      int getInnerMostForDim(statement &s);
      int getForDim(statement &s, const std::string &tag);

      void splitDefineForVariable(varInfo &var);
      void splitDefineAndInitForVariable(varInfo &var);

      void addInnerFors(statement &s);
      void addInnerForsTo(statement &s,
                          varInfoIdMap_t &varInfoIdMap,
                          int &currentInnerID,
                          const int innerDim);

      void checkStatementForExclusives(statement &s,
                                       varInfoIdMap_t &varInfoIdMap,
                                       const int innerID);

      void addOuterFors(statement &s);

      void removeUnnecessaryBlocksInKernel(statement &s);
      void addOccaForsToKernel(statement &s);

      void addOccaFors();

      void setupOccaVariables(statement &s);
    };

    bool isAnOccaID(const std::string &s);
    bool isAnOccaInnerID(const std::string &s);
    bool isAnOccaOuterID(const std::string &s);
    bool isAnOccaGlobalID(const std::string &s);

    bool isAnOccaDim(const std::string &s);
    bool isAnOccaInnerDim(const std::string &s);
    bool isAnOccaOuterDim(const std::string &s);
    bool isAnOccaGlobalDim(const std::string &s);

    strNode* splitContent(const std::string &str, const bool parsingC = true);
    strNode* splitContent(const char *cRoot, const bool parsingC = true);

    bool checkWithLeft(strNode *nodePos,
                       const std::string &leftValue,
                       const std::string &rightValue,
                       const bool parsingC = true);

    void mergeNodeWithLeft(strNode *&nodePos,
                           const bool addSpace = true,
                           const bool parsingC = true);

    strNode* labelCode(strNode *lineNodeRoot, const bool parsingC = true);

    void initKeywords(const bool parsingC = true);
    void initFortranKeywords();
    //==============================================

    //---[ OCCA Loop Info ]-------------
    class occaLoopInfo {
    public:
      statement *sInfo;
      bool parsingC;

      occaLoopInfo();

      occaLoopInfo(statement &s,
                   const bool parsingC_ = true,
                   const std::string &tag = "");

      void lookForLoopFrom(statement &s,
                           const std::string &tag = "");

      void loadForLoopInfo(int &innerDims, int &outerDims,
                           std::string *innerIters,
                           std::string *outerIters);

      void getLoopInfo(std::string &ioLoopVar,
                       std::string &ioLoop,
                       std::string &loopNest);

      void getLoopNode1Info(std::string &iter,
                            std::string &start);

      void getLoopNode2Info(std::string &bound,
                            std::string &iterCheck);

      void getLoopNode3Info(std::string &stride,
                            std::string &strideOpSign,
                            std::string &strideOp);

      void setIterDefaultValues();

      std::string getSetupExpression();
    };
    //==================================
  };

  // Just to ignore the namespace
  class parser : public parserNS::parserBase {};
};

#endif
