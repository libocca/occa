#ifndef OCCA_PARSER_HEADER
#define OCCA_PARSER_HEADER

#include "occaParserDefines.hpp"
#include "occaParserMacro.hpp"
#include "occaParserTools.hpp"
#include "occaParserNodes.hpp"
#include "occaParserTypes.hpp"
#include "occaParserStatement.hpp"

namespace occa {
  namespace parserNamespace {
    class parserBase {
    public:
      static const int version = 100;

      macroMap_t macroMap;
      std::vector<macroInfo> macros;

      bool macrosAreInitialized;

      varOriginMap_t varOriginMap;
      varUsedMap_t varUsedMap;     // Statements are placed backwards

      kernelInfoMap_t kernelInfoMap;

      statement *globalScope;

      parserBase();

      const std::string parseFile(const std::string &filename);
      const std::string parseSource(const char *cRoot);

      //---[ Macro Parser Functions ]-------
      std::string getMacroName(const char *&c);

      bool evaluateMacroStatement(const char *&c);
      static typeHolder evaluateLabelNode(strNode *labelNodeRoot);

      void loadMacroInfo(macroInfo &info, const char *&c);
      int loadMacro(const std::string &line, const int state = doNothing);

      void applyMacros(std::string &line);

      strNode* preprocessMacros(strNode *nodeRoot);

      strNode* splitAndPreprocessContent(const std::string &s);
      strNode* splitAndPreprocessContent(const char *cRoot);
      //====================================

      void initMacros();
      void loadLanguageTypes();

      void applyToAllStatements(statement &s,
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

      bool nodeHasUnknownVariable(strNode *n);

      void setupOccaFors(statement &s);

      bool statementHasOccaOuterFor(statement &s);
      bool statementHasOccaFor(statement &s);

      bool statementHasOklFor(statement &s);

      bool statementHasOccaStuff(statement &s);

      void markKernelFunctions(statement &s);

      void labelKernelsAsNativeOrNot(statement &s);

      void setupCudaVariables(statement &s);

      void addFunctionPrototypes();

      int statementOccaForNest(statement &s);
      bool statementIsAnOccaFor(statement &s);

      void fixOccaForStatementOrder(statement &origin, statementNode *sn);
      void fixOccaForOrder(statement &s);

      void addParallelFors(statement &s);

      void updateConstToConstant(statement &s);

      strNode* occaExclusiveStrNode(varInfo &info,
                                    const int depth,
                                    const int sideDepth);

      void addKernelInfo(varInfo &info, statement &s);

      void addArgQualifiers(varInfo &info, statement &s);

      void modifyExclusiveVariables(statement &s);

      void modifyTextureVariables();

      statementNode* splitKernelStatement(statementNode *sn,
                                          kernelInfo &info);

      void loadKernelInfos();

      void stripOccaFromKernel(statement &s);

      std::string occaScope(statement &s);

      void incrementDepth(statement &s);

      void decrementDepth(statement &s);

      bool statementHasBarrier(statement &s);

      statementNode* findStatementWith(statement &s,
                                       findStatementWith_t func);

      int getKernelOuterDim(statement &s);
      int getKernelInnerDim(statement &s);

      int getOuterMostForDim(statement &s);
      int getInnerMostForDim(statement &s);
      int getForDim(statement &s, const std::string &tag);

      void checkPathForConditionals(statementNode *path);

      int findLoopSections(statement &s,
                           statementNode *path,
                           loopSection_t &loopSection,
                           int section = 0);

      bool varInTwoSegments(varInfo &info,
                            loopSection_t &loopSection);

      varInfoNode* findVarsMovingToTop(statement &s,
                                       loopSection_t &loopSection);

      void splitDefineForVariable(statement *&origin,
                                  varInfo &var, strNode *varNode,
                                  const int declPos);

      void addInnerForsToStatement(statement &s,
                                   const int innerDim);

      statementNode* addInnerForsBetweenBarriers(statement &origin,
                                                 statementNode *includeStart,
                                                 const int innerDim);

      void addInnerFors(statement &s);
      void addOuterFors(statement &s);

      void removeUnnecessaryBlocksInKernel(statement &s);
      void floatSharedVarsInKernel(statement &s);
      void addOccaForsToKernel(statement &s);

      void addOccaFors();

      void setupOccaVariables(statement &s);
    };

    strNode* splitContent(const std::string &str);
    strNode* splitContent(const char *cRoot);

    strNode* labelCode(strNode *lineNodeRoot);

    void initKeywords();
    //==============================================
  };

  // Just to ignore the namespace
  class parser : public parserNamespace::parserBase {};
};

#endif
