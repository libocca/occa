#ifndef OCCA_INTERNAL_LANG_PARSER_HEADER
#define OCCA_INTERNAL_LANG_PARSER_HEADER

#include <map>
#include <vector>

#include <occa/types/json.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>
#include <occa/internal/lang/keyword.hpp>
#include <occa/internal/lang/loaders.hpp>
#include <occa/internal/lang/preprocessor.hpp>
#include <occa/internal/lang/processingStages.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/tokenizer.hpp>
#include <occa/internal/lang/tokenContext.hpp>
#include <occa/internal/lang/statementContext.hpp>
#include <occa/internal/lang/statementPeeker.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    class parser_t;

    typedef stream<token_t*> tokenStream;

    typedef statement_t* (parser_t::*statementLoader_t)(attributeTokenMap &smntAttributes);
    typedef std::map<int, statementLoader_t> statementLoaderMap;

    class parser_t {
     public:
      //---[ Stream ]-------------------
      tokenStream stream;
      tokenizer_t tokenizer;
      preprocessor_t preprocessor;
      stringTokenMerger stringMerger;
      externTokenMerger externMerger;
      newlineTokenFilter newlineFilter;
      unknownTokenFilter unknownFilter;
      //================================

      //---[ Status ]-------------------
      blockStatement root;

      keywords_t keywords;
      statementLoaderMap statementLoaders;
      nameToAttributeMap attributeMap;

      tokenContext_t tokenContext;
      statementContext_t smntContext;
      statementPeeker_t smntPeeker;

      int loadingStatementType;
      bool checkSemicolon;

      unknownToken defaultRootToken;
      statementArray comments;
      attributeTokenMap attributes;

      bool success;
      //================================

      //---[ Misc ]---------------------
      occa::json settings;
      qualifier_t *restrictQualifier;
      //================================

      parser_t(const occa::json &settings_ = occa::json());
      virtual ~parser_t();

      //---[ Customization ]------------
      template <class attributeType>
      void addAttribute();

      virtual void onClear();
      virtual void beforePreprocessing();
      virtual void beforeParsing();
      virtual void afterParsing();
      //================================

      //---[ Public ]-------------------
      virtual bool succeeded() const;

      std::string toString() const;
      void toString(std::string &s) const;

      void writeToFile(const std::string &filename) const;

      void setSourceMetadata(sourceMetadata_t &sourceMetadata) const;
      //================================

      //---[ Setup ]--------------------
      void clear();
      void clearAttributes();
      void clearAttributes(attributeTokenMap &attrs);

      void addSettingDefines();

      void parseSource(const std::string &source);
      void parseFile(const std::string &filename);

      void setSource(const std::string &source,
                     const bool isFile);
      void setupLoadTokens();
      void loadTokens();
      void parseTokens();
      //================================

      //---[ Helper Methods ]-----------
      keyword_t& getKeyword(token_t *token);
      keyword_t& getKeyword(const std::string &name);

      exprNode* parseTokenContextExpression();
      exprNode* parseTokenContextExpression(const int start,
                                            const int end);

      void loadComments();
      void loadComments(const int start,
                        const int end);
      void pushComments();

      void loadAttributes(attributeTokenMap &attrs);

      attribute_t* getAttribute(const std::string &name);

      void addAttributesTo(attributeTokenMap &attrs,
                           statement_t *smnt);

      void loadBaseType(vartype_t &vartype);
      void loadType(vartype_t &vartype);
      vartype_t loadType();

      bool isLoadingVariable();
      bool isLoadingFunction();
      bool isLoadingFunctionPointer();

      void loadVariable(variable_t &var);
      variable_t loadVariable();

      void loadVariable(vartype_t &vartype,
                        variable_t &var);

      void loadFunction(function_t &func);

      int peek();
      //================================

      //---[ Type Loaders ]-------------
      variableDeclaration loadVariableDeclaration(attributeTokenMap &smntAttributes,
                                                  const vartype_t &baseType);

      void applyDeclarationSmntAttributes(attributeTokenMap &smntAttributes,
                                          variable_t &var);

      int declarationNextCheck(const opType_t opCheck);

      void loadDeclarationBitfield(variableDeclaration &decl);

      void loadDeclarationAssignment(variableDeclaration &decl);

      void loadDeclarationBraceInitializer(variableDeclaration &decl);
      //================================

      //---[ Statement Loaders ]--------
      bool isEmpty();

      void loadAllStatements();

      statement_t* loadNextStatement();
      statement_t* getNextStatement();

      statement_t* loadBlockStatement(attributeTokenMap &smntAttributes);

      statement_t* loadEmptyStatement(attributeTokenMap &smntAttributes);

      statement_t* loadExpressionStatement(attributeTokenMap &smntAttributes);

      statement_t* loadDeclarationStatement(attributeTokenMap &smntAttributes);

      statement_t* loadNamespaceStatement(attributeTokenMap &smntAttributes);

      statement_t* loadFunctionStatement(attributeTokenMap &smntAttributes);

      void checkIfConditionStatementExists();
      void loadConditionStatements(statementArray &statements,
                                   const int expectedCount);
      statement_t* loadConditionStatement();

      statement_t* loadIfStatement(attributeTokenMap &smntAttributes);
      statement_t* loadElifStatement(attributeTokenMap &smntAttributes);
      statement_t* loadElseStatement(attributeTokenMap &smntAttributes);

      statement_t* loadForStatement(attributeTokenMap &smntAttributes);
      statement_t* loadWhileStatement(attributeTokenMap &smntAttributes);
      statement_t* loadDoWhileStatement(attributeTokenMap &smntAttributes);

      statement_t* loadSwitchStatement(attributeTokenMap &smntAttributes);
      statement_t* loadCaseStatement(attributeTokenMap &smntAttributes);
      statement_t* loadDefaultStatement(attributeTokenMap &smntAttributes);
      statement_t* loadContinueStatement(attributeTokenMap &smntAttributes);
      statement_t* loadBreakStatement(attributeTokenMap &smntAttributes);

      statement_t* loadReturnStatement(attributeTokenMap &smntAttributes);

      statement_t* loadClassAccessStatement(attributeTokenMap &smntAttributes);

      statement_t* loadDirectiveStatement(attributeTokenMap &smntAttributes);
      statement_t* loadPragmaStatement(attributeTokenMap &smntAttributes);

      statement_t* loadGotoStatement(attributeTokenMap &smntAttributes);
      statement_t* loadGotoLabelStatement(attributeTokenMap &smntAttributes);
      //================================
    };
  }
}

#include "parser.tpp"

#endif
