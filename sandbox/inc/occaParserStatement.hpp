#ifndef OCCA_PARSER_STATEMENT_HEADER
#define OCCA_PARSER_STATEMENT_HEADER

#include "occaParserDefines.hpp"
#include "occaParserMacro.hpp"
#include "occaParserTools.hpp"
#include "occaParserNodes.hpp"
#include "occaParserTypes.hpp"

namespace occa {
  namespace parserNamespace {
    class statement;

    //---[ Exp Node ]-------------------------------
    namespace expType {
      static const int maxBit = 17;

      static const int root            = (1 << 0);

      static const int LCR             = (7 << 1);
      static const int L               = (1 << 1);
      static const int C               = (1 << 2);
      static const int R               = (1 << 3);

      static const int qualifier       = (1 <<  4);
      static const int type            = (1 <<  5);
      static const int presetValue     = (1 <<  6);
      static const int variable        = (1 <<  7);
      static const int function        = (1 <<  8);
      static const int functionPointer = (1 <<  9);
      static const int namespace_      = (1 << 10);
      static const int macro_          = (1 << 11);
      static const int goto_           = (1 << 12);
      static const int gotoLabel_      = (1 << 13);
      static const int case_           = (1 << 14);
      static const int return_         = (1 << 15);
      static const int occaFor         = (1 << 16);

      static const int printValue      = (1 << 17);
    };

    class expNode {
    public:
      statement &sInfo;

      std::string value;
      int info;

      expNode *up;

      int leafCount;
      expNode **leaves;
      varInfo *var;
      typeDef *type;

      expNode(statement &s);
      expNode(expNode &up_);

      //---[ Find Statement ]-----------
      void labelStatement(strNode *&nodeRoot);

      int loadMacroStatement(strNode *&nodeRoot);
      int loadOccaForStatement(strNode *&nodeRoot);
      int loadStructStatement(strNode *&nodeRoot);
      int loadUpdateStatement(strNode *&nodeRoot);
      int loadDescriptorStatement(strNode *&nodeRoot);
      int loadGotoStatement(strNode *&nodeRoot);
      int loadFlowStatement(strNode *&nodeRoot);
      int loadSpecialStatement(strNode *&nodeRoot);
      int loadBlockStatement(strNode *&nodeRoot);
      //================================

      void loadFromNode(strNode *&nodePos);

      void initLoadFromNode(strNode *n);

      void initOrganization();

      void organizeLeaves();
      void organizeLeaves(const int level);

      void addNewVariables(strNode *nodePos);
      void updateNewVariables();

      int mergeRange(const int newLeafType,
                     const int leafPosStart,
                     const int leafPosEnd);

      // [a][::][b]
      void mergeNamespaces();

      // [const] int x
      void mergeQualifiers();

      // [[const] [int] [*]] x
      void mergeTypes();

      // [[[const] [int] [*]] [x]]
      void mergeVariables();

      // 1 [type]                           2 [(]       3 [(]
      // [[qualifiers] [type] [qualifiers]] [(*[name])] [([args])]
      void mergeFunctionPointers();

      // class(...), class{1,2,3}
      void mergeClassConstructs();

      // static_cast<>()
      void mergeCasts();

      // func()
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

      //---[ Custom Functions ]---------
      void labelNewVariables();
      //================================

      //---[ Custom Type Info ]---------
      bool qualifierEndsWithStar() const;

      bool typeEndsWithStar() const;
      //================================

      expNode* clone(statement &s);
      expNode* clone(expNode *original);

      void cloneTo(expNode &newRoot);

      void freeLeaf(const int leafPos);

      void free();

      void print(const std::string &tab = "");

      operator std::string () const;

      friend std::ostream& operator << (std::ostream &out, const expNode &n);
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

      varOriginMap_t &varOriginMap;
      varUsedMap_t   &varUsedMap;

      int depth;
      statement *up;

      int type;

      expNode expRoot;

      // Going away
      strNode *nodeStart, *nodeEnd;

      int statementCount;
      statementNode *statementStart, *statementEnd;

      // Going away
      bool hasTypeDefinition;

      statement(parserBase &pb);

      statement(const int depth_, statement *up_);

      statement(const int depth_,
                const int type_,
                statement *up_,
                strNode *nodeStart_, strNode *nodeEnd_);

      ~statement();

      statement* makeSubStatement();

      std::string getTab() const;

      int statementType(strNode *&nodeRoot);

      int checkMacroStatementType(strNode *&nodeRoot);
      int checkOccaForStatementType(strNode *&nodeRoot);
      int checkStructStatementType(strNode *&nodeRoot);
      int checkUpdateStatementType(strNode *&nodeRoot);
      int checkDescriptorStatementType(strNode *&nodeRoot);
      int checkGotoStatementType(strNode *&nodeRoot);
      int checkFlowStatementType(strNode *&nodeRoot);
      int checkSpecialStatementType(strNode *&nodeRoot);
      int checkBlockStatementType(strNode *&nodeRoot);

      void addTypeDef(const std::string &typeDefName);

      bool nodeHasQualifier(strNode *n) const;
      bool nodeHasSpecifier(strNode *n) const;
      bool nodeHasDescriptor(strNode *n) const;

      varInfo loadVarInfo(strNode *&nodePos);

      typeDef* hasTypeInScope(const std::string &typeName) const;

      varInfo* hasVariableInScope(const std::string &varName) const;

      bool hasDescriptorVariable(const std::string descriptor) const;
      bool hasDescriptorVariableInScope(const std::string descriptor) const;

      void loadAllFromNode(strNode *nodeRoot);
      strNode* loadFromNode(strNode *nodeRoot);

      void loadBlocksFromLastNode(strNode *end,
                                  const int startBlockPos = 0);

      strNode* loadSimpleFromNode(const int st,
                                  strNode *nodeRoot,
                                  strNode *nodeRootEnd);

      strNode* loadForFromNode(const int st,
                               strNode *nodeRoot,
                               strNode *nodeRootEnd);

      strNode* loadWhileFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd);

      strNode* loadIfFromNode(const int st,
                              strNode *nodeRoot,
                              strNode *nodeRootEnd);

      // [-] Missing
      strNode* loadSwitchFromNode(const int st,
                                  strNode *nodeRoot,
                                  strNode *nodeRootEnd);

      strNode* loadGotoFromNode(const int st,
                                strNode *nodeRoot,
                                strNode *nodeRootEnd);

      strNode* loadFunctionDefinitionFromNode(const int st,
                                              strNode *nodeRoot,
                                              strNode *nodeRootEnd);

      strNode* loadFunctionPrototypeFromNode(const int st,
                                             strNode *nodeRoot,
                                             strNode *nodeRootEnd);

      strNode* loadBlockFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd);

      // [-] Missing
      strNode* loadStructFromNode(const int st,
                                  strNode *nodeRoot,
                                  strNode *nodeRootEnd);

      // [-] Missing
      strNode* loadBlankFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd);

      // [-] Missing
      strNode* loadMacroFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd);

      varInfo* addVariable(const varInfo &info,
                           statement *origin = NULL);

      void addStatement(statement *newStatement);

      statement* clone();

      void printVariablesInStatement();

      void printVariablesInScope();

      void printTypesInStatement();
      void printTypeDefsInStatement();

      void printTypesInScope();

      // autoMode: Handles newlines and tabs
      std::string prettyString(strNode *nodeRoot,
                               const std::string &tab_ = "",
                               const bool autoMode = true) const;

      operator std::string() const;
    };

    std::ostream& operator << (std::ostream &out, const statement &s);
  };
};

#endif
