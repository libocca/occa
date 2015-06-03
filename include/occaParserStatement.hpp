#ifndef OCCA_PARSER_STATEMENT_HEADER
#define OCCA_PARSER_STATEMENT_HEADER

#include "occaParserDefines.hpp"
#include "occaParserMacro.hpp"
#include "occaParserTools.hpp"
#include "occaParserNodes.hpp"
#include "occaParserTypes.hpp"

namespace occa {
  namespace parserNS {
    class statement;
    class sDep_t;

    //---[ Exp Node ]-------------------------------
    namespace expType {
      static const int root            = 0;

      static const int LCR             = (7 << 1);
      static const int LR              = (5 << 1);

      static const int L               = (1 << 1);
      static const int C               = (1 << 2);
      static const int R               = (1 << 3);

      static const int qualifier       = (1 <<  4);
      static const int type            = (1 <<  5);
      static const int presetValue     = (1 <<  6);
      static const int operator_       = (1 <<  7);
      static const int unknown         = (1 <<  8);
      static const int variable        = (1 <<  9);
      static const int function        = (1 << 11);
      static const int functionPointer = (1 << 12);
      static const int typedef_        = (1 << 13);
      static const int prototype       = (1 << 14);
      static const int declaration     = (1 << 15);
      static const int struct_         = (1 << 16);
      static const int namespace_      = (1 << 17);
      static const int cast_           = (1 << 18);
      static const int macro_          = (1 << 19);
      static const int goto_           = (1 << 20);
      static const int gotoLabel_      = (1 << 21);
      static const int return_         = (1 << 22);
      static const int transfer_       = (1 << 23);
      static const int occaFor         = (1 << 24);
      static const int checkSInfo      = (1 << 25);
      static const int attribute       = (1 << 26);

      static const int hasInfo         = (3 << 27);
      static const int varInfo         = (1 << 27);
      static const int typeInfo        = (1 << 28);

      static const int printValue      = (1 << 29);
      static const int printLeaves     = (1 << 30);
      static const int maxBit          = 31;
    };

    namespace expFlag {
      static const int none        = 0;
      static const int noNewline   = (1 << 0);
      static const int noSemicolon = (1 << 1);

      static const int addVarToScope  = (1 << 0);
      static const int addTypeToScope = (1 << 1);
      static const int addToParent    = (1 << 2);
    };

    namespace leafType {
      static const char exp  = (1 << 0);
      static const char type = (1 << 1);
      static const char var  = (1 << 2);
    };

    namespace statementFlag {
      static const int updateByNumber     = (1 << 0);
      static const int updateByUnderscore = (1 << 1);

      static const int printEverything    = (int) -1;
      static const int printSubStatements = (1 << 0);
    };

    class varInfo;
    class typeInfo;
    class expNode;

    class varLeaf_t {
    public:
      bool hasExp;

      varInfo *var;
      expNode *exp;

      varLeaf_t() :
        hasExp(false),

        var(NULL),
        exp(NULL) {}
    };

    class expNode {
    public:
      statement *sInfo;

      std::string value;
      int info;

      expNode *up;

      int leafCount;
      expNode **leaves;

      expNode();
      expNode(statement &s);
      expNode(expNode &up_);

      bool operator == (expNode &e);

      fnvOutput_t hash();
      bool sameAs(expNode &e, const bool nestedSearch = true);

      inline expNode& operator [] (const int i){
        if(0 <= i)
          return *leaves[i];
        else
          return *leaves[leafCount + i];
      }

      void loadFromNode(strNode *&nodePos, const bool parsingC = true);

      void splitAndOrganizeNode(strNode *nodeRoot);
      void splitAndOrganizeFortranNode(strNode *nodeRoot);

      void organize(const bool parsingC = true);

      void splitDeclareStatement(const int flags = (expFlag::addVarToScope |
                                                    expFlag::addToParent));

      void splitFlowStatement();

      void splitFunctionStatement(const int flags = (expFlag::addVarToScope |
                                                     expFlag::addToParent));

      void splitStructStatement(const int flags = (expFlag::addTypeToScope |
                                                   expFlag::addToParent));

      void splitCaseStatement(const bool parsingC = true);

      //  ---[ Fortran ]------
      void splitFortranDeclareStatement();
      void splitFortranUpdateStatement();
      void splitFortranFlowStatement();
      void splitFortranForStatement();
      void splitFortranFunctionStatement();
      //  ====================

      static void translateOccaKeyword(strNode *nodePos, const bool parsingC);

      void initLoadFromNode(strNode *nodeRoot);
      void initLoadFromFortranNode(strNode *nodeRoot);

      void initOrganization();

      void organizeLeaves(const bool inRoot = true);
      void organizeFortranLeaves();

      void organizeLeaves(const int level);

      int mergeRange(const int newLeafType,
                     const int leafPosStart,
                     const int leafPosEnd);

      // [a][::][b]
      void mergeNamespaces();

      // [(class)]
      void labelCasts();

      // const int [*] x
      void labelReferenceQualifiers();

      // Add used vars to varUsedMap
      void labelUsedVariables();

      // @(attributes)
      void loadAttributes();

      // <const int,float>
      void mergeTypes();

      // class(...), class{1,2,3}
      void mergeClassConstructs();

      // static_cast<>()
      void mergeCasts();

      // [max(a,b)]
      void mergeFunctionCalls();

      void mergeArguments();

      // a[3]
      void mergeArrays();

      // ptr(a,b)
      void mergePointerArrays();

      // (class) x
      void mergeClassCasts();

      // sizeof x
      void mergeSizeOf();

      // new, new [], delete, delete []
      void mergeNewsAndDeletes();

      // throw x
      void mergeThrows();

      // [++]i
      int mergeLeftUnary(const int leafPos, const bool leftToRight);

      // i[++]
      int mergeRightUnary(const int leafPos, const bool leftToRight);

      // a [+] b
      int mergeBinary(const int leafPos, const bool leftToRight);

      // a [?] b : c
      int mergeTernary(const int leafPos, const bool leftToRight);

      //---[ Custom Type Info ]---------
      bool qualifierEndsWithStar();

      bool typeEndsWithStar();

      bool hasAnArrayQualifier(const int pos = 0);

      void mergeFortranArrays();

      void translateFortranMemberCalls();
      void translateFortranPow();
      //================================

      static void swap(expNode &a, expNode &b);

      expNode* clone();
      expNode* clone(statement &s);
      expNode* clone(expNode *original);

      void cloneTo(expNode &newExp);

      expNode* lastLeaf();

      //---[ Exp Info ]-----------------
      int depth();
      int whichLeafAmI();
      int nestedLeafCount();

      expNode* makeFlatHandle();
      void makeFlatHandle(int &offset,
                          expNode **flatLeaves);

      static void freeFlatHandle(expNode &flatRoot);

      expNode* makeCsvFlatHandle();

      void addNode(const int info_ = 0, const int pos = -1);
      void addNode(const int info_, const std::string &value_, const int pos = -1);
      void addNodes(const int info_, const int pos_, const int count = 1);

      void addNode(expNode &node_, const int pos_ = -1);

      void reserve(const int count);
      void reserveAndShift(const int pos, const int count = 1);

      void setLeaf(expNode &leaf, const int pos);

      varInfo& addVarInfoNode();
      varInfo& addVarInfoNode(const int pos_);

      void putVarInfo(varInfo &var);
      void putVarInfo(const int pos, varInfo &var);

      typeInfo& addTypeInfoNode();
      typeInfo& addTypeInfoNode(const int pos);

      bool hasVariable();

      varInfo& getVarInfo();
      varInfo& getVarInfo(const int pos_);

      void setVarInfo(varInfo &var);
      void setVarInfo(const int pos_, varInfo &var);

      typeInfo& getTypeInfo();
      typeInfo& getTypeInfo(const int pos);

      void removeNodes(const int pos, const int count = 1);
      void removeNode(const int pos = 0);

      bool hasQualifier(const std::string &qualifier);

      void removeQualifier(const std::string &qualifier);

      void changeType(const std::string &newType);

      int getVariableCount();
      bool variableHasInit(const int pos);

      expNode* getVariableNode(const int pos);
      expNode* getVariableInfoNode(const int pos);
      expNode* getVariableOpNode(const int pos);
      expNode* getVariableInitNode(const int pos);

      std::string getVariableName(const int pos = 0);

      int getUpdatedVariableCount();
      bool updatedVariableIsSet(const int pos);

      expNode* getUpdatedNode(const int pos);
      expNode* getUpdatedVariableInfoNode(const int pos);
      expNode* getUpdatedVariableOpNode(const int pos);
      expNode* getUpdatedVariableSetNode(const int pos);

      int getVariableBracketCount();
      expNode* getVariableBracket(const int pos);

      //  ---[ Node-based ]--------
      std::string getMyVariableName();
      //  =========================

      //  ---[ Statement-based ]---
      void switchBaseStatement(statement &s1, statement &s2);
      //  =========================
      //================================


      //---[ Analysis Info ]------------
      bool valueIsKnown(const strToStrMap_t &stsMap = strToStrMap_t());
      typeHolder calculateValue(const strToStrMap_t &stsMap = strToStrMap_t()); // Assumes (valueIsKnown() == true)
      //================================

      void freeLeaf(const int leafPos);
      void free();
      void freeThis();

      void print(const std::string &tab = "");
      void printOn(std::ostream &out,
                   const std::string &tab = "",
                   const int flags = expFlag::none);

      static void printVec(expVec_t &v);

      std::string toString(const std::string &tab = "");
      std::string toString(const int leafPos, const int printLeafCount);

      operator std::string ();

      friend std::ostream& operator << (std::ostream &out, expNode &n);
    };

    struct statementExp {
      int sType;
      expNode exp;
    };
    //==============================================


    //---[ Statement ]------------------------------
    class statement {
    public:
      scopeTypeMap_t scopeTypeMap;
      scopeVarMap_t scopeVarMap;

      varUsedMap_t &varUpdateMap;
      varUsedMap_t &varUsedMap;

      int depth;
      int info;

      statement *up;

      expNode expRoot;

      int statementCount;
      statementNode *statementStart, *statementEnd;

      attributeMap_t attributeMap;

      statement(parserBase &pb);

      statement(const int depth_,
                varUsedMap_t &varUpdateMap_,
                varUsedMap_t &varUsedMap_);

      statement(const int depth_, statement *up_);

      statement(const int depth_,
                const int type_,
                statement *up_);

      ~statement();

      statement& operator [] (const int snPos);
      statement& operator [] (intVector_t &path);

      int getSubIndex();

      int getDepth();
      void setIndexPath(intVector_t &path, statement *target = NULL);

      statement* makeSubStatement();

      std::string getTab();

      //---[ Find Statement ]-----------
      void labelStatement(strNode *&nodeRoot,
                          expNode *expPtr = NULL,
                          const bool parsingC = true);

      int findStatementType(strNode *&nodeRoot,
                            expNode *expPtr = NULL,
                          const bool parsingC = true);

      int findFortranStatementType(strNode *&nodeRoot,
                                   expNode *expPtr = NULL);

      int checkMacroStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkOccaForStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkStructStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkUpdateStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkDescriptorStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkGotoStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkFlowStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkSpecialStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkBlockStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);

      //  ---[ Fortran ]------
      int checkFortranStructStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkFortranUpdateStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkFortranDescriptorStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkFortranFlowStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);
      int checkFortranSpecialStatementType(strNode *&nodeRoot, expNode *expPtr = NULL);

      int getStatementType();
      //================================

      void addType(typeInfo &type);
      void addTypedef(const std::string &typedefName);

      bool nodeHasQualifier(strNode *n);
      bool nodeHasSpecifier(strNode *n);
      bool nodeHasDescriptor(strNode *n);

      typeInfo* hasTypeInScope(const std::string &typeName);
      typeInfo* hasTypeInLocalScope(const std::string &typeName);

      varInfo* hasVariableInScope(const std::string &varName);
      varInfo* hasVariableInLocalScope(const std::string &varName);

      bool hasDescriptorVariable(const std::string descriptor);
      bool hasDescriptorVariableInScope(const std::string descriptor);

      //---[ Loading ]------------------
      void loadAllFromNode(strNode *nodeRoot, const bool parsingC = true);

      strNode* loadFromNode(strNode *nodeRoot, const bool parsingC = true);

      void setExpNodeFromStrNode(expNode &exp,
                                 strNode *nodePos);

      expNode* createExpNodeFrom(strNode *nodePos);
      expNode* createExpNodeFrom(const std::string &source);

      expNode* createPlainExpNodeFrom(const std::string &source);

      strNode* loadSimpleFromNode(const int st,
                                  strNode *nodeRoot,
                                  strNode *nodeRootEnd,
                                  const bool parsingC = true);

      strNode* loadOneStatementFromNode(const int st,
                                        strNode *nodeRoot,
                                        strNode *nodeRootEnd,
                                        const bool parsingC = true);

      strNode* loadForFromNode(const int st,
                               strNode *nodeRoot,
                               strNode *nodeRootEnd,
                               const bool parsingC = true);

      strNode* loadWhileFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd,
                                 const bool parsingC = true);

      strNode* loadIfFromNode(const int st,
                              strNode *nodeRoot,
                              strNode *nodeRootEnd,
                              const bool parsingC = true);

      strNode* loadSwitchFromNode(const int st,
                                  strNode *nodeRoot,
                                  strNode *nodeRootEnd,
                                  const bool parsingC = true);

      strNode* loadCaseFromNode(const int st,
                                strNode *nodeRoot,
                                strNode *nodeRootEnd,
                                const bool parsingC = true);

      strNode* loadGotoFromNode(const int st,
                                strNode *nodeRoot,
                                strNode *nodeRootEnd,
                                const bool parsingC = true);

      strNode* loadFunctionDefinitionFromNode(const int st,
                                              strNode *nodeRoot,
                                              strNode *nodeRootEnd,
                                              const bool parsingC = true);

      strNode* loadFunctionPrototypeFromNode(const int st,
                                             strNode *nodeRoot,
                                             strNode *nodeRootEnd,
                                             const bool parsingC = true);

      strNode* loadBlockFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd,
                                 const bool parsingC = true);

      // [-] Missing
      strNode* loadStructFromNode(const int st,
                                  strNode *nodeRoot,
                                  strNode *nodeRootEnd,
                                  const bool parsingC = true);

      // [-] Missing
      strNode* loadBlankFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd,
                                 const bool parsingC = true);

      // [-] Missing
      strNode* loadMacroFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd,
                                 const bool parsingC = true);

      bool isFortranEnd(strNode *nodePos);
      strNode* getFortranEnd(strNode *nodePos);
      static strNode* getFortranEnd(strNode *nodePos,
                                    const std::string &value);

      strNode* loadUntilFortranEnd(strNode *nodePos);

      static strNode* skipNodeUntil(strNode *nodePos,
                                    const std::string &value,
                                    int *separation = NULL);

      static strNode* skipAfterStatement(strNode *nodePos);
      static strNode* skipUntilStatementEnd(strNode *nodePos);

      static strNode* skipUntilFortranStatementEnd(strNode *nodePos);
      //================================

      statement* getGlobalScope();
      statementNode* getStatementNode();

      statement& pushNewStatementLeft(const int type_ = 0);
      statement& pushNewStatementRight(const int type_ = 0);

      statement& createStatementFromSource(const std::string &source);

      void addStatementFromSource(const std::string &source);
      void addStatementsFromSource(const std::string &source);

      void pushSourceLeftOf(statementNode *target,
                            const std::string &source);

      void pushSourceRightOf(statementNode *target,
                             const std::string &source);

      //---[ Misc ]---------------------
      bool hasBarrier();
      bool hasStatementWithBarrier();

      // Guaranteed to work with statements under a globalScope
      statement& greatestCommonStatement(statement &s);

      unsigned int distToForLoop();
      unsigned int distToOccaForLoop();
      unsigned int distToStatementType(const int info_);

      bool insideOf(statement &s);

      void setStatementIdMap(statementIdMap_t &idMap);

      void setStatementIdMap(statementIdMap_t &idMap,
                             int &startID);

      void setStatementVector(statementVector_t &vec,
                              const bool init = true);

      static void setStatementVector(statementIdMap_t &idMap,
                                     statementVector_t &vec);

      void removeFromUpdateMapFor(varInfo &var);
      void removeFromUsedMapFor(varInfo &var);
      void removeFromMapFor(varInfo &var,
                            varUsedMap_t &usedMap);
      //================================

      void checkIfVariableIsDefined(varInfo &var,
                                    statement *origin);

      statement* getVarOriginStatement(varInfo &var);

      // Add from stack memory
      varInfo& addVariable(varInfo &var,
                           statement *origin = NULL);

      // Add from pre-allocated memory
      void addVariable(varInfo *var,
                       statement *origin = NULL);

      void addVariableToUpdateMap(varInfo &var,
                                  statement *origin_ = NULL);

      void addVariableToUsedMap(varInfo &var,
                                statement *origin_ = NULL);

      void addVariableToMap(varInfo &var,
                            varUsedMap_t &usedMap,
                            statement *origin);

      void addStatement(statement *newStatement);
      void removeStatement(statement &s);

      statement* clone(statement *up_ = NULL);

      void printVariablesInScope();
      void printVariablesInLocalScope();

      void printTypesInScope();
      void printTypesInStatement();
      void printTypeDefsInStatement();

      //---[ Statement Info ]-----------
      void createUniqueVariables(std::vector<std::string> &names,
                                 const int flags = statementFlag::updateByNumber);

      void createUniqueSequentialVariables(std::string &varName,
                                           const int varCount);

      void swapExpWith(statement &s);

      bool hasQualifier(const std::string &qualifier);
      void addQualifier(const std::string &qualifier, const int pos = 0);
      void removeQualifier(const std::string &qualifier);

      int occaForInfo();

      static int occaForNest(const int forInfo);
      static bool isOccaOuterFor(const int forInfo);
      static bool isOccaInnerFor(const int forInfo);

      void setVariableDeps(varInfo &var,
                           sDep_t &sDep);

      void addVariableDeps(expNode &exp,
                           sDep_t &sDep);

      bool setsVariableValue(varInfo &var);

      void addStatementDependencies(statementIdMap_t &idMap,
                                    statementVector_t sVec,
                                    idDepMap_t &depMap);

      void addStatementDependencies(statement &fromS,
                                    statementIdMap_t &idMap,
                                    statementVector_t sVec,
                                    idDepMap_t &depMap);

      void addNestedDependencies(statementIdMap_t &idMap,
                                 statementVector_t sVec,
                                 idDepMap_t &depMap);

      varInfo& getDeclarationVarInfo(const int pos);
      expNode* getDeclarationVarNode(const int pos);
      std::string getDeclarationVarName(const int pos);
      expNode* getDeclarationVarInitNode(const int pos);
      int getDeclarationVarCount();

      varInfo* getFunctionVar();
      void setFunctionVar(varInfo &var);
      std::string getFunctionName();
      void setFunctionName(const std::string &newName);
      bool functionHasQualifier(const std::string &qName);

      int getFunctionArgCount();
      std::string getFunctionArgType(const int pos);
      std::string getFunctionArgName(const int pos);
      varInfo* getFunctionArgVar(const int pos);
      bool hasFunctionArgVar(varInfo &var);

      void addFunctionArg(const int pos, varInfo &var);

      expNode* getForStatement(const int pos);
      void addForStatement();
      int getForStatementCount();
      //================================

      // autoMode: Handles newlines and tabs
      std::string prettyString(strNode *nodeRoot,
                               const std::string &tab_ = "",
                               const bool autoMode = true);

      std::string toString(const int flags = (statementFlag::printSubStatements));
      std::string onlyThisToString();

      operator std::string();
    };

    std::ostream& operator << (std::ostream &out, statement &s);
    //============================================

    bool isAnOccaTag(const std::string &tag);
    bool isAnOccaInnerTag(const std::string &tag);
    bool isAnOccaOuterTag(const std::string &tag);
  };
};

#endif
