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

#ifndef OCCA_PARSER_STATEMENT_HEADER
#define OCCA_PARSER_STATEMENT_HEADER

#include "occa/defines.hpp"
#include "occa/parser/defines.hpp"
#include "occa/parser/preprocessor.hpp"
#include "occa/parser/tools.hpp"
#include "occa/parser/nodes.hpp"
#include "occa/parser/types.hpp"
#include "occa/tools/hash.hpp"

namespace occa {
  namespace parserNS {
    class statement;
    class sDep_t;

    //---[ Exp Node ]-------------------------------
    namespace expType {
      static const info_t root            = 0;
      static const info_t firstPass       = (((info_t) 1)  << 63);
      static const info_t firstPassMask   = ~firstPass;

      static const info_t L               = (((info_t) 1)  << 0);
      static const info_t R               = (((info_t) 1)  << 1);
      static const info_t L_R             = (((info_t) 3)  << 0);

      static const info_t LR              = (((info_t) 1)  << 2);
      static const info_t C               = (((info_t) 1)  << 3);
      static const info_t LCR             = (((info_t) 1)  << 4);
      static const info_t operator_       = (L | R | LR | C | LCR);

      static const info_t descriptor      = (((info_t) 7)  <<  5);
      static const info_t qualifier       = (((info_t) 1)  <<  5);
      static const info_t type            = (((info_t) 1)  <<  6);
      static const info_t struct_         = (((info_t) 1)  <<  7);
      static const info_t presetValue     = (((info_t) 1)  <<  8);
      static const info_t endStatement    = (((info_t) 1)  <<  9);
      static const info_t unknown         = (((info_t) 1)  << 10);
      static const info_t variable        = (((info_t) 1)  << 11);
      static const info_t function        = (((info_t) 1)  << 12);
      static const info_t prototype       = (((info_t) 1)  << 13);
      static const info_t declaration     = (((info_t) 1)  << 14);
      static const info_t namespace_      = (((info_t) 1)  << 15);
      static const info_t cast_           = (((info_t) 1)  << 16);
      static const info_t macro_          = (((info_t) 1)  << 17);
      static const info_t goto_           = (((info_t) 1)  << 18);
      static const info_t gotoLabel_      = (((info_t) 1)  << 19);
      static const info_t return_         = (((info_t) 1)  << 20);
      static const info_t transfer_       = (((info_t) 1)  << 21);
      static const info_t occaFor         = (((info_t) 1)  << 22);
      static const info_t checkSInfo      = (((info_t) 1)  << 23);
      static const info_t attribute       = (((info_t) 1)  << 24);

      static const info_t hasInfo         = (((info_t) 7)  << 25);
      static const info_t varInfo         = (((info_t) 1)  << 25);
      static const info_t typeInfo        = (((info_t) 1)  << 26);

      static const info_t asm_            = (((info_t) 1)  << 27);

      static const info_t printValue      = (((info_t) 1)  << 28);
      static const info_t printLeaves     = (((info_t) 1)  << 29);

      static const info_t flowControl     = (((info_t) 1)  << 30);

      static const info_t specialKeyword  = (((info_t) 1)  << 31);

      static const info_t macroKeyword    = (((info_t) 1)  << 32);

      static const info_t apiKeyword      = (((info_t) 7)  << 33);
      static const info_t occaKeyword     = (((info_t) 1)  << 33);
      static const info_t cudaKeyword     = (((info_t) 1)  << 34);
      static const info_t openclKeyword   = (((info_t) 1)  << 35);

      static const info_t hasFlag         = (((info_t) 1)  << 36);
      static const info_t removeFlags     = ~expType::hasFlag;
      static const info_t hasSemicolon    = (((info_t) 1)  << 36);
    }

    namespace expFlag {
      static const info_t none         = 0;
      static const info_t noNewline    = (1 << 0);
      static const info_t noSemicolon  = (1 << 1);
      static const info_t endWithComma = (1 << 2);

      static const info_t addVarToScope  = (1 << 0);
      static const info_t addTypeToScope = (1 << 1);
      static const info_t addToParent    = (1 << 2);
    }

    namespace statementFlag {
      static const info_t updateByNumber     = (1 << 0);
      static const info_t updateByUnderscore = (1 << 1);

      static const info_t printEverything    = (info_t) -1;
      static const info_t printSubStatements = (1 << 0);
    }

    class expNode {
    public:
      statement *sInfo;

      std::string value;
      info_t info;

      expNode *up;

      int leafCount;
      expNode **leaves;

      expNode();
      expNode(const char *c);
      expNode(const std::string &str);
      expNode(const expNode &e);
      expNode(statement &s);

      expNode& operator = (const expNode &e);

      bool operator == (expNode &e);

      hash_t hash();
      bool sameAs(expNode &e, const bool nestedSearch = true);

      inline expNode& operator [] (const int i) {
        if (0 <= i) {
          return *leaves[i];
        } else {
          return *leaves[leafCount + i];
        }
      }

      expNode makeFloatingLeaf();

      void loadFromNode(expNode &allExp,
                        const int parsingLanguage_ = parserInfo::parsingC);
      void loadFromNode(expNode &allExp,
                        int &expPos,
                        const int parsingLanguage_);

      void loadAttributes();
      void loadAttributes(expNode &allExp,
                          int &expPos);

      void organizeNode();
      void organizeFortranNode();

      void organize(const int parsingLanguage_ = parserInfo::parsingC);

      void organizeDeclareStatement(const info_t flags = (expFlag::addVarToScope |
                                                          expFlag::addToParent));

      void organizeUpdateStatement();

      void organizeFlowStatement();

      void organizeFunctionStatement();

      void organizeStructStatement();

      void organizeCaseStatement(const int parsingLanguage_ = parserInfo::parsingC);

      //  ---[ Fortran ]------
      void organizeFortranDeclareStatement();
      void organizeFortranUpdateStatement();
      void organizeFortranFlowStatement();
      void organizeFortranForStatement();
      void organizeFortranFunctionStatement();
      //  ====================

      static void translateOccaKeyword(expNode &exp,
                                       info_t preInfo,
                                       const int parsingLanguage_ = parserInfo::parsingC);

      bool isOrganized();
      bool needsExpChange();
      void changeExpTypes(const int leafPos = 0);

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

      // <const int,float>
      void mergeTypes();

      // a[3]
      void mergeArrays();

      // class(...), class{1,2,3}
      void mergeClassConstructs();

      // static_cast<>()
      void mergeCasts();

      // [max(a,b)]
      void mergeFunctionCalls();

      void mergeArguments();

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

      void subtractOneFrom(expNode &e);

      void translateFortranMemberCalls();
      void translateFortranPow();
      //================================

      static void swap(expNode &a, expNode &b);

      expNode clone();
      expNode clone(statement &s);

      expNode* clonePtr();
      expNode* clonePtr(statement &s);

      void cloneTo(expNode &newExp);

      expNode* lastLeaf();

      //---[ Exp Info ]-----------------
      int depth();
      int whichLeafAmI();
      int nestedLeafCount();

      expNode& lastNode();

      expNode* makeDumbFlatHandle();
      void makeDumbFlatHandle(int &offset,
                              expNode **flatLeaves);

      expNode* makeFlatHandle();
      void makeFlatHandle(int &offset,
                          expNode **flatLeaves);

      static void freeFlatHandle(expNode &flatRoot);

      expNode* makeCsvFlatHandle();

      void addNode(const info_t info_ = 0, const int pos = -1);
      void addNode(const info_t info_, const std::string &value_, const int pos = -1);
      void addNode(expNode &node_, const int pos_ = -1);

      void addNodes(const int count);
      void addNodes(const int pos_, const int count);
      void addNodes(const info_t info_, const int pos_, const int count);

      void reserve(const int count);
      int insertExpAt(expNode &exp, int pos);
      void useExpLeaves(expNode &exp, const int pos, const int count);
      void copyAndUseExpLeaves(expNode &exp, const int pos, const int count);
      void reserveAndShift(const int pos, const int count = 1);

      void setLeaf(expNode &leaf, const int pos);

      void removeNodes(const int pos, const int count = 1);
      void removeNode(const int pos = -1);

      template <class TM>
      TM& addInfoNode() {
        addNode(0);

        TM **tmLeaves = (TM**) leaves;
        TM *&tmLeaf   = tmLeaves[0];

        tmLeaf = new TM();
        return *tmLeaf;
      }

      template <class TM>
      TM& addInfoNode(const info_t info_, const int pos_) {
        const int pos = ((0 <= pos_) ? pos_ : leafCount);

        addNode(info_, pos);

        return leaves[pos]->addInfoNode<TM>();
      }

      template <class TM>
      void putInfo(const info_t info_, TM &t) {
        addNode(0);
        leaves[0] = (expNode*) &t;

        info = info_;
      }

      template <class TM>
      void putInfo(const info_t info_, const int pos_, TM &t) {
        const int pos = ((0 <= pos_) ? pos_ : leafCount);

        addNode(info_, pos);
        leaves[pos]->putInfo<TM>(info_, t);
      }

      template <class TM>
      TM& getInfo() {
        return *((TM*) leaves[0]);
      }

      template <class TM>
      TM& getInfo(const int pos_) {
        const int pos = ((0 <= pos_) ? pos_ : leafCount);

        TM **tmLeaves = (TM**) leaves[pos]->leaves;
        TM *&tmLeaf   = tmLeaves[0];

        return *tmLeaf;
      }

      template <class TM>
      void setInfo(TM &tm) {
        leaves[0] = (expNode*) &tm;
      }

      template <class TM>
      void setInfo(const int pos_, TM &tm) {
        const int pos = ((0 <= pos_) ? pos_ : leafCount);

        TM **tmLeaves = (TM**) leaves[pos]->leaves;
        TM *&tmLeaf   = tmLeaves[0];

        tmLeaf = &tm;
      }

      // typeInfo
      typeInfo& addTypeInfoNode();
      typeInfo& addTypeInfoNode(const int pos);

      void putTypeInfo(typeInfo &type);
      void putTypeInfo(const int pos, typeInfo &type);

      typeInfo& getTypeInfo();
      typeInfo& getTypeInfo(const int pos);

      void setTypeInfo(typeInfo &type);
      void setTypeInfo(const int pos, typeInfo &type);

      // varInfo
      varInfo& addVarInfoNode();
      varInfo& addVarInfoNode(const int pos);

      void putVarInfo(varInfo &var);
      void putVarInfo(const int pos, varInfo &var);

      varInfo& getVarInfo();
      varInfo& getVarInfo(const int pos);

      void setVarInfo(varInfo &var);
      void setVarInfo(const int pos, varInfo &var);

      bool hasVariable();

      varInfo typeInfoOf(const std::string &str);

      varInfo evaluateType();

      bool hasQualifier(const std::string &qualifier);

      void removeQualifier(const std::string &qualifier);

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
      void setNestedSInfo(statement *sInfo_);
      void setNestedSInfo(statement &sInfo_);
      //  =========================
      //================================

      //---[ Analysis Info ]------------
      bool valueIsKnown(const strToStrMap_t &stsMap = strToStrMap_t());
      typeHolder calculateValue(const strToStrMap_t &stsMap = strToStrMap_t());
      //================================

      void freeLeaf(const int leafPos);
      void free();
      void freeThis();

      void print(const std::string &tab = "");

      void printOnString(std::string &str,
                         const std::string &tab = "",
                         const info_t flags = expFlag::none);

      static void printVec(expVector_t &v);

      inline std::string toString(const std::string &tab = "",
                                  const info_t flags = expFlag::none) {
        std::string ret;
        printOnString(ret, tab, flags);
        return ret;
      }

      inline operator std::string () {
        std::string ret;
        printOnString(ret);
        return ret;
      }

      friend std::ostream& operator << (std::ostream &out, expNode &n);
    };
    //==============================================


    //---[ Statement ]------------------------------
    namespace smntType {
      static const info_t invalidStatement   = (1 << 0);

      static const info_t simpleStatement    = (7 << 1);
      static const info_t typedefStatement   = (1 << 1);
      static const info_t declareStatement   = (1 << 2);
      static const info_t updateStatement    = (1 << 3);

      static const info_t flowStatement      = (0xFF <<  4);
      static const info_t forStatement       = (1    <<  4);
      static const info_t whileStatement     = (1    <<  5);
      static const info_t doWhileStatement   = (3    <<  5);
      static const info_t ifStatement        = (1    <<  7);
      static const info_t elseIfStatement    = (3    <<  7);
      static const info_t elseStatement      = (5    <<  7);
      static const info_t switchStatement    = (1    << 10);
      static const info_t gotoStatement      = (1    << 11);

      static const info_t caseStatement      = (1 << 12);
      static const info_t namespaceStatement = (1 << 13);
      static const info_t blankStatement     = (1 << 14);

      static const info_t functionStatement  = (3 << 15);
      static const info_t functionDefinition = (1 << 15);
      static const info_t functionPrototype  = (1 << 16);
      static const info_t blockStatement     = (1 << 17);
      static const info_t structStatement    = (1 << 18);

      static const info_t occaStatement      = (1 << 19);
      static const info_t occaFor            = (occaStatement |
                                                forStatement);

      static const info_t macroStatement     = (1 << 20);
      static const info_t skipStatement      = (1 << 21);
    }

    class statement {
    public:
      parserBase &parser;
      scopeInfo *scope;

      info_t info;

      statement *up;

      expNode expRoot;

      statementNode *statementStart, *statementEnd;

      attributeMap_t attributeMap;

      statement(parserBase &pb);
      statement(const statement &s);

      statement(statement *up_);

      statement(const info_t info_,
                statement *up_);

      ~statement();

      statement& operator [] (const int snPos);
      statement& operator [] (intVector_t &path);

      int getSubIndex();

      int depth();
      int statementCount();

      void setIndexPath(intVector_t &path, statement *target = NULL);

      statement* makeSubStatement();

      std::string getTab();

      //---[ Find Statement ]---------------------
      void labelStatement(expNode &allExp,
                          int &expPos,
                          const int parsingLanguage_ = parserInfo::parsingC);

      info_t findStatementType(expNode &allExp,
                               int &expPos,
                               const int parsingLanguage_ = parserInfo::parsingC);

      info_t findFortranStatementType(expNode &allExp,
                                      int &expPos);

      info_t checkMacroStatementType(expNode &allExp, int &expPos);
      info_t checkOccaForStatementType(expNode &allExp, int &expPos);
      info_t checkStructStatementType(expNode &allExp, int &expPos);
      info_t checkUpdateStatementType(expNode &allExp, int &expPos);
      info_t checkDescriptorStatementType(expNode &allExp, int &expPos);
      info_t checkGotoStatementType(expNode &allExp, int &expPos);
      info_t checkFlowStatementType(expNode &allExp, int &expPos);
      info_t checkNamespaceStatementType(expNode &allExp, int &expPos);
      info_t checkSpecialStatementType(expNode &allExp, int &expPos);
      info_t checkBlockStatementType(expNode &allExp, int &expPos);

      //  ---[ Fortran ]----------------
      info_t checkFortranStructStatementType(expNode &allExp, int &expPos);
      info_t checkFortranUpdateStatementType(expNode &allExp, int &expPos);
      info_t checkFortranDescriptorStatementType(expNode &allExp, int &expPos);
      info_t checkFortranFlowStatementType(expNode &allExp, int &expPos);
      info_t checkFortranSpecialStatementType(expNode &allExp, int &expPos);
      //==========================================

      //  ---[ Attributes ]-------------
      attribute_t& attribute(const std::string &attr);
      attribute_t* hasAttribute(const std::string &attr);
      void addAttribute(attribute_t &attr);
      void addAttribute(const std::string &attrSource);
      void addAttributeTag(const std::string &attrName);
      void removeAttribute(const std::string &attr);

      std::string attributeMapToString();
      void printAttributeMap();

      void updateInitialLoopAttributes();

      void updateOccaOMLoopAttributes(const std::string &loopTag,
                                      const std::string &loopNest);
      //================================

      void addType(typeInfo &type);
      void addTypedef(const std::string &typedefName);

      bool expHasSpecifier(expNode &allExp, int expPos);
      bool expHasDescriptor(expNode &allExp, int expPos);

      typeInfo* hasTypeInScope(const std::string &typeName);
      typeInfo* hasTypeInLocalScope(const std::string &typeName);

      varInfo* hasVariableInScope(const std::string &varName);
      varInfo* hasVariableInLocalScope(const std::string &varName);

      bool hasDescriptorVariable(const std::string descriptor);
      bool hasDescriptorVariableInScope(const std::string descriptor);

      void removeFromScope(typeInfo &type);
      void removeFromScope(varInfo &var);

      void removeTypeFromScope(const std::string &typeName);
      void removeVarFromScope(const std::string &varName);

      //---[ Loading ]------------------
      void loadAllFromNode(expNode allExp, const int parsingLanguage_ = parserInfo::parsingC);

      void loadFromNode(expNode allExp);

      void loadFromNode(expNode allExp,
                        const int parsingLanguage_);

      void loadFromNode(expNode &allExp,
                        int &expPos,
                        const int parsingLanguage_);

      static expNode createExpNodeFrom(const std::string &source);
      expNode createPlainExpNodeFrom(const std::string &source);

      void reloadFromSource(const std::string &source);

      expNode createOrganizedExpNodeFrom(const std::string &source);
      expNode createOrganizedExpNodeFrom(expNode &allExp,
                                         const int expPos,
                                         const int leafCount);

      void loadSimpleFromNode(const info_t st,
                              expNode &allExp,
                              int &expPos,
                              const int parsingLanguage_ = parserInfo::parsingC);

      void loadOneStatementFromNode(const info_t st,
                                    expNode &allExp,
                                    int &expPos,
                                    const int parsingLanguage_ = parserInfo::parsingC);

      void loadForFromNode(const info_t st,
                           expNode &allExp,
                           int &expPos,
                           const int parsingLanguage_ = parserInfo::parsingC);

      void loadWhileFromNode(const info_t st,
                             expNode &allExp,
                             int &expPos,
                             const int parsingLanguage_ = parserInfo::parsingC);

      void loadIfFromNode(const info_t st,
                          expNode &allExp,
                          int &expPos,
                          const int parsingLanguage_ = parserInfo::parsingC);

      void loadSwitchFromNode(const info_t st,
                              expNode &allExp,
                              int &expPos,
                              const int parsingLanguage_ = parserInfo::parsingC);

      void loadGotoFromNode(const info_t st,
                            expNode &allExp,
                            int &expPos,
                            const int parsingLanguage_ = parserInfo::parsingC);

      void loadCaseFromNode(const info_t st,
                            expNode &allExp,
                            int &expPos,
                            const int parsingLanguage_ = parserInfo::parsingC);

      void loadFunctionDefinitionFromNode(const info_t st,
                                          expNode &allExp,
                                          int &expPos,
                                          const int parsingLanguage_ = parserInfo::parsingC);

      void loadFunctionPrototypeFromNode(const info_t st,
                                         expNode &allExp,
                                         int &expPos,
                                         const int parsingLanguage_ = parserInfo::parsingC);

      void loadBlockFromNode(const info_t st,
                             expNode &allExp,
                             int &expPos,
                             const int parsingLanguage_ = parserInfo::parsingC);

      void loadNamespaceFromNode(const info_t st,
                                 expNode &allExp,
                                 int &expPos,
                                 const int parsingLanguage_ = parserInfo::parsingC);

      // [-] Missing
      void loadStructFromNode(const info_t st,
                              expNode &allExp,
                              int &expPos,
                              const int parsingLanguage_ = parserInfo::parsingC);

      // [-] Missing
      void loadBlankFromNode(const info_t st,
                             expNode &allExp,
                             int &expPos,
                             const int parsingLanguage_ = parserInfo::parsingC);

      // [-] Missing
      void loadMacroFromNode(const info_t st,
                             expNode &allExp,
                             int &expPos,
                             const int parsingLanguage_ = parserInfo::parsingC);

      bool isFortranEnd(expNode &allExp, int &expPos);

      void loadUntilFortranEnd(expNode &allExp, int &expPos);

      static void skipAfterStatement(expNode &allExp, int &expPos);
      static void skipUntilStatementEnd(expNode &allExp, int &expPos);

      static void skipUntilFortranStatementEnd(expNode &allExp, int &expPos);
      //================================

      statement* getGlobalScope();
      scopeInfo* getNamespace();

      statementNode* getStatementNode();

      void pushLastStatementLeftOf(statement *target);
      void pushLastStatementRightOf(statement *target);

      void pushLeftOf(statement *target, statement *s);
      void pushRightOf(statement *target, statement *s);

      statement& pushNewStatementLeft(const info_t info_ = 0);
      statement& pushNewStatementRight(const info_t info_ = 0);

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
      unsigned int distToStatementType(const info_t info_);

      bool insideOf(statement &s);

      void setStatementIdMap(statementIdMap_t &idMap);

      void setStatementIdMap(statementIdMap_t &idMap,
                             int &startID);

      void setStatementVector(statementVector_t &vec,
                              const bool init = true);

      static void setStatementVector(statementIdMap_t &idMap,
                                     statementVector_t &vec);
      //================================

      void checkIfVariableIsDefined(varInfo &var,
                                    statement *origin);

      // Add from stack memory
      varInfo& addVariable(varInfo &var,
                           statement *origin = NULL);

      // Add from pre-allocated memory
      void addVariable(varInfo *var,
                       statement *origin_ = NULL);

      // Swap variable varInfo*
      void replaceVarInfos(varToVarMap_t &v2v);

      void addStatement(statement *newStatement);
      void removeStatement(statement &s);

      static void swap(statement &a, statement &b);
      static void swapPlaces(statement &a, statement &b);
      static void swapStatementNodesFor(statement &a, statement &b);

      statement* clone(statement *up_ = NULL);

      void printVariablesInScope();
      void printVariablesInLocalScope();

      void printTypesInScope();
      void printTypesInStatement();

      //---[ Statement Info ]-----------
      void createUniqueVariables(std::vector<std::string> &names,
                                 const info_t flags = statementFlag::updateByNumber);

      void createUniqueSequentialVariables(std::string &varName,
                                           const int varCount);

      void swapExpWith(statement &s);

      bool hasQualifier(const std::string &qualifier);
      void addQualifier(const std::string &qualifier, const int pos = 0);
      void removeQualifier(const std::string &qualifier);

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

      void printDebugInfo();

      void printOnString(std::string &str,
                         const info_t flags = (statementFlag::printSubStatements));

      void printSubsOnString(std::string &str);

      inline std::string toString(const info_t flags = (statementFlag::printSubStatements)) {
        std::string ret;
        printOnString(ret, flags);
        return ret;
      }

      inline std::string onlyThisToString() {
        std::string ret;
        printOnString(ret, (statementFlag::printEverything &
                            ~statementFlag::printSubStatements));
        return ret;
      }

      inline operator std::string () {
        std::string ret;
        printOnString(ret);
        return ret;
      }
    };

    std::ostream& operator << (std::ostream &out, statement &s);
    //============================================

    bool isAnOccaTag(const std::string &tag);
    bool isAnOccaInnerTag(const std::string &tag);
    bool isAnOccaOuterTag(const std::string &tag);
  }
}

#endif
