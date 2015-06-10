#ifndef OCCA_PARSER_HEADER
#define OCCA_PARSER_HEADER

#include "occaParserDefines.hpp"
#include "occaParserPreprocessor.hpp"
#include "occaParserTools.hpp"
#include "occaParserNodes.hpp"
#include "occaParserTypes.hpp"
#include "occaParserStatement.hpp"
#include "occaParserMagic.hpp"
#include "occaTools.hpp"

namespace occa {
  namespace parserNS {
    class occaLoopInfo;

    extern intVector_t loadedLanguageVec;

    int loadedLanguage();

    void pushLanguage(const int language);
    int popLanguage();

    class parserBase {
    public:
      std::string filename;

      int parsingLanguage;

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
                                  const int parsingLanguage_ = parserInfo::parsingC);

      const std::string parseSource(const char *cRoot);

      //---[ Macro Parser Functions ]---
      std::string getMacroName(const char *&c);
      std::string getMacroIncludeFile(const char *&c);

      typeHolder evaluateMacroStatement(const char *c);
      bool evaluateMacroBoolStatement(const char *c);

      void loadMacroInfo(macroInfo &info, const char *&c);

      int loadMacro(expNode &allExp, int leafPos, const int state = doNothing);
      int loadMacro(const std::string &line, const int state = doNothing);
      int loadMacro(expNode &allExp, int leafPos, const std::string &line, const int state = doNothing);

      void applyMacros(std::string &line);

      void preprocessMacros(expNode &allExp);

      expNode splitAndPreprocessContent(const std::string &s,
                                        const int parsingLanguage_ = parserInfo::parsingC);
      expNode splitAndPreprocessContent(const char *cRoot,
                                        const int parsingLanguage_ = parserInfo::parsingC);
      //====================================

      void initMacros(const int parsingLanguage_ = parserInfo::parsingC);
      void initFortranMacros();

      void loadLanguageTypes();

      void applyToAllStatements(statement &s,
                                applyToAllStatements_t func);

      void applyToAllKernels(statement &s,
                             applyToAllStatements_t func);

      void applyToStatementsDefiningVar(applyToStatementsDefiningVar_t func);

      void applyToStatementsUsingVar(varInfo &info,
                                     applyToStatementsUsingVar_t func);

      static bool statementIsAKernel(statement &s);

      static statement* getStatementKernel(statement &s);
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

      void markKernelFunctions(statement &s);

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

      //---[ Operator Information ]---------------
      varInfo* hasOperator(const info_t expInfo,
                           const std::string &op,
                           varInfo &var);

      varInfo* hasOperator(const info_t expInfo,
                           const std::string &op,
                           varInfo &varL,
                           varInfo &varR);

      varInfo thVarInfo(const info_t thType);

      varInfo thOperatorReturnType(const info_t expInfo,
                                   const std::string &op,
                                   const info_t thType);

      varInfo thOperatorReturnType(const info_t expInfo,
                                   const std::string &op,
                                   const info_t thTypeL,
                                   const info_t thTypeR);
      //==========================================
    };

    bool isAnOccaID(const std::string &s);
    bool isAnOccaInnerID(const std::string &s);
    bool isAnOccaOuterID(const std::string &s);
    bool isAnOccaGlobalID(const std::string &s);

    bool isAnOccaDim(const std::string &s);
    bool isAnOccaInnerDim(const std::string &s);
    bool isAnOccaOuterDim(const std::string &s);
    bool isAnOccaGlobalDim(const std::string &s);

    expNode splitContent(const std::string &str,
                         const int parsingLanguage_ = parserInfo::parsingC);
    expNode splitContent(const char *cRoot,
                         const int parsingLanguage_ = parserInfo::parsingC);

    expNode splitAndLabelContent(const std::string &str,
                                 const int parsingLanguage_ = parserInfo::parsingC);
    expNode splitAndLabelContent(const char *cRoot,
                                 const int parsingLanguage_ = parserInfo::parsingC);
    expNode splitAndOrganizeContent(const std::string &str,
                                    const int parsingLanguage_ = parserInfo::parsingC);
    expNode splitAndOrganizeContent(const char *cRoot,
                                    const int parsingLanguage_ = parserInfo::parsingC);

    expNode& labelCode(expNode &allExp,
                       const int parsingLanguage_ = parserInfo::parsingC);

    bool checkLastTwoNodes(expNode &node,
                           const std::string &leftValue,
                           const std::string &rightValue,
                           const int parsingLanguage_ = parserInfo::parsingC);

    void mergeLastTwoNodes(expNode &node,
                           const bool addSpace = true,
                           const int parsingLanguage_ = parserInfo::parsingC);

    expNode createExpNodeFrom(const std::string &source);

    void loadKeywords(const int parsingLanguage_ = parserInfo::parsingC);
    void loadCKeywords();
    void loadFortranKeywords();
    void initCKeywords();
    void initFortranKeywords();

    //---[ OCCA Loop Info ]-------------
    class occaLoopInfo {
    public:
      statement *sInfo;
      int parsingLanguage;

      occaLoopInfo(statement &s,
                   const int parsingLanguage_ = parserInfo::parsingC,
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
    //==============================================
  };

  // Just to ignore the namespace
  class parser : public parserNS::parserBase {};
};

#endif
