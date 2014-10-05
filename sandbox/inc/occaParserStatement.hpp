#ifndef OCCA_PARSER_STATEMENT_HEADER
#define OCCA_PARSER_STATEMENT_HEADER

#include "occaParserDefines.hpp"
#include "occaParserMacro.hpp"
#include "occaParserTools.hpp"
#include "occaParserNodes.hpp"
#include "occaParserTypes.hpp"

namespace occa {
  namespace parserNamespace {
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

      strNode *nodeStart, *nodeEnd;

      int statementCount;
      statementNode *statementStart, *statementEnd;

      bool hasTypeDefinition;

      statement(parserBase &pb);

      statement(const int depth_,
                const int type_,
                statement *up_,
                strNode *nodeStart_, strNode *nodeEnd_);

      ~statement();

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

      statement* clone() const;

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
