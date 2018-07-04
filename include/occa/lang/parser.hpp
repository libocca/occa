/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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
#ifndef OCCA_LANG_PARSER_HEADER
#define OCCA_LANG_PARSER_HEADER

#include <list>
#include <map>
#include <vector>

#include <occa/tools/properties.hpp>
#include <occa/lang/exprTransform.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/keyword.hpp>
#include <occa/lang/preprocessor.hpp>
#include <occa/lang/processingStages.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/statementTransform.hpp>
#include <occa/lang/tokenizer.hpp>
#include <occa/lang/tokenContext.hpp>
#include <occa/lang/variable.hpp>

namespace occa {
  namespace lang {
    class parser_t;

    typedef stream<token_t*>   tokenStream;
    typedef std::map<int, int> keywordToStatementMap;

    typedef statement_t* (parser_t::*statementLoader_t)(attributeTokenMap &smntAttributes);
    typedef std::map<int, statementLoader_t> statementLoaderMap;

    typedef std::list<blockStatement*>          blockStatementList;
    typedef std::map<std::string, attribute_t*> nameToAttributeMap;

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
      tokenContext context;
      keywordMap keywords;
      keywordToStatementMap keywordPeek;
      statementLoaderMap statementLoaders;
      nameToAttributeMap attributeMap;

      int lastPeek;
      int lastPeekPosition;
      bool checkSemicolon;

      unknownToken defaultRootToken;
      blockStatement root;
      blockStatement *up;
      blockStatementList upStack;
      attributeTokenMap attributes;

      bool success;
      //================================

      //---[ Misc ]---------------------
      occa::properties settings;
      //================================

      parser_t(const occa::properties &settings_ = occa::properties());
      virtual ~parser_t();

      //---[ Public ]-------------------
      virtual bool succeeded() const;

      std::string toString() const;
      void toString(std::string &s) const;

      void writeToFile(const std::string &filename) const;

      void setMetadata(kernelMetadataMap &metadataMap) const;
      //================================

      //---[ Setup ]--------------------
      void clear();
      void clearAttributes();
      void clearAttributes(attributeTokenMap &attrs);

      void addSettingDefines();

      void pushUp(blockStatement &newUp);
      void popUp();

      void parseSource(const std::string &source);
      void parseFile(const std::string &filename);

      void setSource(const std::string &source,
                     const bool isFile);
      void loadTokens();
      void parseTokens();

      keyword_t& getKeyword(token_t *token);
      opType_t getOperatorType(token_t *token);
      //================================

      //---[ Helper Methods ]-----------
      exprNode* getExpression();
      exprNode* getExpression(const int start,
                              const int end);
      token_t* replaceIdentifier(identifierToken &identifier);

      attribute_t* getAttribute(const std::string &name);

      void loadAttributes(attributeTokenMap &attrs);
      void loadAttribute(attributeTokenMap &attrs);
      void setAttributeArgs(attributeToken_t &attr,
                            tokenRangeVector &argRanges);
      void addAttributesTo(attributeTokenMap &attrs,
                           statement_t *smnt);
      //================================

      //---[ Peek ]---------------------
      int peek();
      int uncachedPeek();

      void setupPeek();

      int peekIdentifier(const int tokenIndex);
      bool isGotoLabel(const int tokenIndex);

      int peekOperator(const int tokenIndex);
      //================================

      //---[ Type Loaders ]-------------
      variable_t loadVariable();

      variableDeclaration loadVariableDeclaration(attributeTokenMap &smntAttributes,
                                                  const vartype_t &baseType);
      void loadDeclarationAttributes(attributeTokenMap &smntAttributes,
                                     variableDeclaration &decl);
      int declarationNextCheck(const opType_t opCheck);
      void loadDeclarationBitfield(variableDeclaration &decl);
      void loadDeclarationAssignment(variableDeclaration &decl);
      void loadDeclarationBraceInitializer(variableDeclaration &decl);

      vartype_t loadType();

      void loadBaseType(vartype_t &vartype);

      void loadQualifier(token_t *token,
                         const qualifier_t &qualifier,
                         vartype_t &vartype);

      void setPointers(vartype_t &vartype);
      void setPointer(vartype_t &vartype);

      void setReference(vartype_t &vartype);

      bool isLoadingFunctionPointer();
      bool isLoadingVariable();
      bool isLoadingFunction();

      variable_t loadFunctionPointer(vartype_t &vartype);
      variable_t loadVariable(vartype_t &vartype);

      bool hasArray();
      void setArrays(vartype_t &vartype);
      void setArray(vartype_t &vartype);

      void setArguments(functionPtr_t &func);
      void setArguments(function_t &func);

    private:
      template <class funcType>
      void setArgumentsFor(funcType &func);

    public:
      void getArgumentRanges(tokenRangeVector &argRanges);
      variable_t getArgument();

      class_t loadClassType();
      struct_t loadStructType();
      enum_t loadEnumType();
      union_t loadUnionType();
      //================================

      //---[ Loader Helpers ]-----------
      bool isEmpty();
      statement_t* getNextStatement();
      //================================

      //---[ Statement Loaders ]--------
      void loadAllStatements();

      statement_t* loadBlockStatement(attributeTokenMap &smntAttributes);

      statement_t* loadEmptyStatement(attributeTokenMap &smntAttributes);

      statement_t* loadExpressionStatement(attributeTokenMap &smntAttributes);

      statement_t* loadDeclarationStatement(attributeTokenMap &smntAttributes);

      statement_t* loadNamespaceStatement(attributeTokenMap &smntAttributes);

      statement_t* loadTypeDeclStatement(attributeTokenMap &smntAttributes);

      statement_t* loadFunctionStatement(attributeTokenMap &smntAttributes);

      void checkIfConditionStatementExists();
      void loadConditionStatements(statementPtrVector &statements,
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

      statement_t* loadPragmaStatement(attributeTokenMap &smntAttributes);

      statement_t* loadGotoStatement(attributeTokenMap &smntAttributes);
      statement_t* loadGotoLabelStatement(attributeTokenMap &smntAttributes);
      //================================

      //---[ Customization ]------------
      template <class attributeType>
      void addAttribute();

      virtual void onClear();
      virtual void beforePreprocessing();
      virtual void beforeParsing();
      virtual void afterParsing();
      //================================
    };
  }
}

#include <occa/lang/parser.tpp>

#endif
