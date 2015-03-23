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
    namespace preExpType {
      static const int descriptor       = (15 << 0);
      static const int typedef_         = (1  << 0); // typedef
      static const int struct_          = (1  << 1); // struct, class (Same as [ s ])
      static const int specifier        = (1  << 2); // void, char, short, int
      static const int qualifier        = (1  << 3); // const, restrict, volatile

      static const int operator_        = (0x1F << 4);
      static const int unitaryOperator  = (3    << 4);
      static const int lUnitaryOperator = (1    << 4);
      static const int rUnitaryOperator = (1    << 5);
      static const int binaryOperator   = (3    << 6);
      static const int assOperator      = (1    << 7); // hehe
      static const int ternaryOperator  = (1    << 8);

      static const int parentheses      = (1 <<  9);
      static const int brace            = (1 << 10);
      static const int bracket          = (1 << 11);

      static const int startSection     = (1 << 12);
      static const int endSection       = (1 << 13);

      static const int startParentheses = (parentheses | startSection);
      static const int endParentheses   = (parentheses | endSection);

      static const int startBrace       = (brace | startSection);
      static const int endBrace         = (brace | endSection);

      static const int startBracket     = (bracket | startSection);
      static const int endBracket       = (bracket | endSection);

      static const int endStatement     = (1 << 14);

      static const int flowControl      = (1 << 15);

      static const int presetValue      = (1 << 16);
      static const int unknownVariable  = (1 << 17);

      static const int specialKeyword   = (1 << 18);

      static const int macroKeyword     = (1 << 19);

      static const int apiKeyword       = (7 << 20);
      static const int occaKeyword      = (1 << 20);
      static const int cudaKeyword      = (1 << 21);
      static const int openclKeyword    = (1 << 22);
    };

    namespace expType {
      static const int root            = (1 << 0);

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
      static const int functionPointer = (1 << 10);
      static const int typedef_        = (1 << 11);
      static const int prototype       = (1 << 12);
      static const int declaration     = (1 << 13);
      static const int struct_         = (1 << 14);
      static const int namespace_      = (1 << 15);
      static const int cast_           = (1 << 16);
      static const int macro_          = (1 << 17);
      static const int goto_           = (1 << 18);
      static const int gotoLabel_      = (1 << 19);
      static const int return_         = (1 << 20);
      static const int transfer_       = (1 << 21);
      static const int occaFor         = (1 << 22);
      static const int checkSInfo      = (1 << 23);

      static const int hasInfo         = (3 << 24);
      static const int varInfo         = (1 << 24);
      static const int typeInfo        = (1 << 25);
      static const int funcInfo        = (1 << 26);

      static const int printValue      = (1 << 27);
      static const int printLeaves     = (1 << 28);
      static const int maxBit          = 29;
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
    };

    class expNode {
    public:
      statement *sInfo;

      std::string value;
      int info;

      expNode *up;

      int leafCount;
      expNode **leaves;

      typeInfo *type;

      expNode();
      expNode(const char *c);
      expNode(const std::string &str);
      expNode(const expNode &e);
      expNode(statement &s);

      expNode& operator = (const expNode &e);

      expNode makeFloatingLeaf();

      inline expNode& operator [] (const int i){
        if(0 <= i)
          return *leaves[i];
        else
          return *leaves[leafCount + i];
      }

      //---[ Find Statement ]-----------
      int getStatementType();
      //================================

      void loadFromNode(expNode &allExp, const bool parsingC = true);
      void loadFromNode(expNode &allExp, int &expPos, const bool parsingC = true);

      void splitAndOrganizeNode();
      void splitAndOrganizeFortranNode();

      void organize(const bool parsingC = true);

      void splitDeclareStatement(const int flags = (expFlag::addVarToScope |
                                                    expFlag::addToParent));

      void splitUpdateStatement();

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

      static void translateOccaKeyword(expNode &exp, const bool parsingC);

      void changeExpTypes();
      void changeFortranExpTypes();

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

      // (class) x
      void mergeClassCasts();

      // sizeof x
      void mergeSizeOf();

      // new, new [], delete, delete []
      void mergeNewsAndDeletes();

      // throw x
      void mergeThrows();

      // [++]i
      int mergeLeftUnary(const int leafPos);

      // i[++]
      int mergeRightUnary(const int leafPos);

      // a [+] b
      int mergeBinary(const int leafPos);

      // a [?] b : c
      int mergeTernary(const int leafPos);

      //---[ Custom Type Info ]---------
      bool qualifierEndsWithStar();

      bool typeEndsWithStar();

      bool hasAnArrayQualifier(const int pos = 0);

      void mergeFortranArrays();

      void translateFortranMemberCalls();
      void translateFortranPow();
      //================================

      static void swap(expNode &a, expNode &b);

      expNode clone();
      expNode clone(statement &s);
      void cloneTo(expNode &newExp);

      expNode* lastLeaf();

      //---[ Exp Info ]-----------------
      int depth();
      int whichLeafAmI();
      int nestedLeafCount();

      expNode& lastNode();

      expNode* makeFlatHandle();
      void makeFlatHandle(int &offset,
                          expNode **flatLeaves);

      static void freeFlatHandle(expNode &flatRoot);

      void addNode(const int info_ = 0, const int pos = -1);
      void addNode(const int info_, const std::string &value_, const int pos = -1);
      void addNodes(const int info_, const int pos_, const int count = 1);

      void addNode(expNode &node_, const int pos_ = -1);

      int insertExpAt(expNode &exp, int pos);
      void useExpLeaves(expNode &exp, const int pos, const int count);
      void reserveAndShift(const int pos, const int count = 1);

      varInfo& addVarInfoNode();
      varInfo& addVarInfoNode(const int pos);

      void putVarInfo(varInfo &var);
      void putVarInfo(const int pos, varInfo &var);

      typeInfo& addTypeInfoNode();
      typeInfo& addTypeInfoNode(const int pos);

      bool hasVariable();

      varInfo& getVarInfo();
      varInfo& getVarInfo(const int pos);

      void setVarInfo(varInfo &var);
      void setVarInfo(const int pos, varInfo &var);

      typeInfo& getTypeInfo();
      typeInfo& getTypeInfo(const int pos);

      void removeNodes(const int pos, const int count = 1);
      void removeNode(const int pos = -1);

      void convertTo(const int info_ = 0);

      bool hasQualifier(const std::string &qualifier);

      void addQualifier(const std::string &qualifier, const int pos = 0);
      void addPostQualifier(const std::string &qualifier, const int pos = 0);

      void removeQualifier(const std::string &qualifier);

      void changeType(const std::string &newType);

      int getVariableCount();
      bool variableHasInit(const int pos);

      expNode* getVariableNode(const int pos);
      expNode* getVariableInfoNode(const int pos);
      expNode* getVariableInitNode(const int pos);
      expNode* getVariableRhsNode(const int pos);

      std::string getVariableName(const int pos = 0);

      //  ---[ Node-based ]--------
      std::string getMyVariableName();
      //  =========================

      //  ---[ Statement-based ]---
      void setNestedSInfo(statement &sInfo_);
      void switchBaseStatement(statement &s1, statement &s2);
      //  =========================
      //================================


      //---[ Analysis Info ]------------
      bool valueIsKnown(const strToStrMap_t &stsMap = strToStrMap_t());
      typeHolder calculateValue(const strToStrMap_t &stsMap = strToStrMap_t());
      //================================

      void freeLeaf(const int leafPos);
      void free();

      void print(const std::string &tab = "");
      void printOn(std::ostream &out,
                   const std::string &tab = "",
                   const int flags = expFlag::none);

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
    namespace smntType {
      static const int invalidStatement   = (1 << 0);

      static const int simpleStatement    = (7 << 1);
      static const int typedefStatement   = (1 << 1);
      static const int declareStatement   = (1 << 2);
      static const int updateStatement    = (1 << 3);

      static const int flowStatement      = (255 <<  4);
      static const int forStatement       = (1   <<  4);
      static const int whileStatement     = (1   <<  5);
      static const int doWhileStatement   = (3   <<  5);
      static const int ifStatement        = (1   <<  7);
      static const int elseIfStatement    = (3   <<  7);
      static const int elseStatement      = (5   <<  7);
      static const int switchStatement    = (1   << 10);
      static const int gotoStatement      = (1   << 11);

      static const int caseStatement      = (1 << 12);
      static const int blankStatement     = (1 << 13);

      static const int functionStatement  = (3 << 14);
      static const int functionDefinition = (1 << 14);
      static const int functionPrototype  = (1 << 15);
      static const int blockStatement     = (1 << 16);
      static const int structStatement    = (1 << 17);

      static const int occaStatement      = (1 << 18);
      static const int occaFor            = (occaStatement |
                                             forStatement);

      static const int macroStatement     = (1 << 19);
      static const int skipStatement      = (1 << 20);
    };

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

      statement(parserBase &pb);

      statement(const int depth_,
                varUsedMap_t &varUpdateMap_,
                varUsedMap_t &varUsedMap_);

      statement(const int depth_, statement *up_);

      statement(const int depth_,
                const int type_,
                statement *up_);

      ~statement();

      statement* makeSubStatement();

      std::string getTab();

      //---[ Find Statement ]-----------
      void labelStatement(expNode &allExp,
                          int &expPos,
                          const bool parsingC = true);

      int findStatementType(expNode &allExp,
                            int &expPos,
                            const bool parsingC = true);

      int findFortranStatementType(expNode &allExp,
                                   int &expPos);

      int checkMacroStatementType(expNode &allExp, int &expPos);
      int checkOccaForStatementType(expNode &allExp, int &expPos);
      int checkStructStatementType(expNode &allExp, int &expPos);
      int checkUpdateStatementType(expNode &allExp, int &expPos);
      int checkDescriptorStatementType(expNode &allExp, int &expPos);
      int checkGotoStatementType(expNode &allExp, int &expPos);
      int checkFlowStatementType(expNode &allExp, int &expPos);
      int checkSpecialStatementType(expNode &allExp, int &expPos);
      int checkBlockStatementType(expNode &allExp, int &expPos);

      //  ---[ Fortran ]------
      int checkFortranStructStatementType(expNode &allExp, int &expPos);
      int checkFortranUpdateStatementType(expNode &allExp, int &expPos);
      int checkFortranDescriptorStatementType(expNode &allExp, int &expPos);
      int checkFortranFlowStatementType(expNode &allExp, int &expPos);
      int checkFortranSpecialStatementType(expNode &allExp, int &expPos);

      int getStatementType();
      //================================

      void addType(typeInfo &type);
      void addTypedef(const std::string &typedefName);

      bool expHasQualifier(expNode &allExp, int expPos);
      bool expHasSpecifier(expNode &allExp, int expPos);
      bool expHasDescriptor(expNode &allExp, int expPos);

      typeInfo* hasTypeInScope(const std::string &typeName);
      typeInfo* hasTypeInLocalScope(const std::string &typeName);

      varInfo* hasVariableInScope(const std::string &varName);
      varInfo* hasVariableInLocalScope(const std::string &varName);

      bool hasDescriptorVariable(const std::string descriptor);
      bool hasDescriptorVariableInScope(const std::string descriptor);

      //---[ Loading ]------------------
      void loadAllFromNode(expNode allExp, const bool parsingC = true);

      void loadFromNode(expNode allExp,
                        const bool parsingC = true);

      void loadFromNode(expNode &allExp,
                        int &expPos,
                        const bool parsingC = true);

      expNode createExpNodeFrom(const std::string &source);
      expNode createPlainExpNodeFrom(const std::string &source);

      void loadSimpleFromNode(const int st,
                              expNode &allExp,
                              int &expPos,
                              const bool parsingC = true);

      void loadOneStatementFromNode(const int st,
                                    expNode &allExp,
                                    int &expPos,
                                    const bool parsingC = true);

      void loadForFromNode(const int st,
                           expNode &allExp,
                           int &expPos,
                           const bool parsingC = true);

      void loadWhileFromNode(const int st,
                             expNode &allExp,
                             int &expPos,
                             const bool parsingC = true);

      void loadIfFromNode(const int st,
                          expNode &allExp,
                          int &expPos,
                          const bool parsingC = true);

      void loadSwitchFromNode(const int st,
                              expNode &allExp,
                              int &expPos,
                              const bool parsingC = true);

      void loadCaseFromNode(const int st,
                            expNode &allExp,
                            int &expPos,
                            const bool parsingC = true);

      void loadGotoFromNode(const int st,
                            expNode &allExp,
                            int &expPos,
                            const bool parsingC = true);

      void loadFunctionDefinitionFromNode(const int st,
                                          expNode &allExp,
                                          int &expPos,
                                          const bool parsingC = true);

      void loadFunctionPrototypeFromNode(const int st,
                                         expNode &allExp,
                                         int &expPos,
                                         const bool parsingC = true);

      void loadBlockFromNode(const int st,
                             expNode &allExp,
                             int &expPos,
                             const bool parsingC = true);

      // [-] Missing
      void loadStructFromNode(const int st,
                              expNode &allExp,
                              int &expPos,
                              const bool parsingC = true);

      // [-] Missing
      void loadBlankFromNode(const int st,
                             expNode &allExp,
                             int &expPos,
                             const bool parsingC = true);

      // [-] Missing
      void loadMacroFromNode(const int st,
                             expNode &allExp,
                             int &expPos,
                             const bool parsingC = true);

      bool isFortranEnd(expNode &allExp, int &expPos);

      void loadUntilFortranEnd(expNode &allExp, int &expPos);

      static void skipAfterStatement(expNode &allExp, int &expPos);
      static void skipUntilStatementEnd(expNode &allExp, int &expPos);

      static void skipUntilFortranStatementEnd(expNode &allExp, int &expPos);
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

      bool guaranteesBreak();

      unsigned int distToForLoop();
      unsigned int distToOccaForLoop();
      unsigned int distToStatementType(const int info_);

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
      int getForStatementCount();
      //================================

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
