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

#include "occa/tools/properties.hpp"
#include "exprTransform.hpp"
#include "keyword.hpp"
#include "preprocessor.hpp"
#include "processingStages.hpp"
#include "statement.hpp"
#include "statementTransform.hpp"
#include "tokenizer.hpp"
#include "tokenContext.hpp"
#include "variable.hpp"

namespace occa {
  namespace lang {
    class parser_t;

    typedef stream<token_t*>   tokenStream;
    typedef std::map<int, int> keywordToStatementMap;

    typedef statement_t* (parser_t::*statementLoader_t)();
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
      ~parser_t();

      //---[ Setup ]--------------------
      void clear();
      void clearAttributes();
      void clearAttributes(attributeTokenMap &attrs);

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

      variableDeclaration loadVariableDeclaration(const vartype_t &baseType);
      void loadDeclarationAttributes(variableDeclaration &decl);
      int declarationNextCheck(const opType_t opCheck);
      void loadDeclarationBitfield(variableDeclaration &decl);
      void loadDeclarationAssignment(variableDeclaration &decl);
      void loadDeclarationBraceInitializer(variableDeclaration &decl);

      vartype_t preloadType();

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

      statement_t* loadBlockStatement();

      statement_t* loadEmptyStatement();

      statement_t* loadExpressionStatement();

      statement_t* loadDeclarationStatement();

      statement_t* loadNamespaceStatement();

      statement_t* loadTypeDeclStatement();

      statement_t* loadFunctionStatement();

      void checkIfConditionStatementExists();
      void loadConditionStatements(statementPtrVector &statements,
                                   const int expectedCount);
      statement_t* loadConditionStatement();

      statement_t* loadIfStatement();
      statement_t* loadElifStatement();
      statement_t* loadElseStatement();

      statement_t* loadForStatement();
      statement_t* loadWhileStatement();
      statement_t* loadDoWhileStatement();

      statement_t* loadSwitchStatement();
      statement_t* loadCaseStatement();
      statement_t* loadDefaultStatement();
      statement_t* loadContinueStatement();
      statement_t* loadBreakStatement();

      statement_t* loadReturnStatement();

      statement_t* loadClassAccessStatement();

      statement_t* loadPragmaStatement();

      statement_t* loadGotoStatement();
      statement_t* loadGotoLabelStatement();
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

#include "parser.tpp"

#endif
