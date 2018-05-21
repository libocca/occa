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

#include "keyword.hpp"
#include "preprocessor.hpp"
#include "processingStages.hpp"
#include "statement.hpp"
#include "tokenizer.hpp"
#include "tokenContext.hpp"
#include "variable.hpp"

namespace occa {
  namespace lang {
    class parser_t;

    typedef stream<token_t*>   tokenStream;
    typedef std::map<int, int> keywordToStatementMap;

    typedef statement_t* (parser_t::*statementLoader_t)(blockStatement &blockSmnt);
    typedef std::map<int, statementLoader_t> statementLoaderMap;

    typedef std::map<std::string, attribute_t*> nameToAttributeMap;

    class parser_t {
    public:
      //---[ Stream ]-------------------
      tokenStream stream;
      tokenizer_t tokenizer;
      preprocessor_t preprocessor;
      stringTokenMerger stringMerger;
      newlineTokenMerger newlineMerger;
      unknownTokenFilter unknownFilter;
      //================================

      //---[ Status ]-------------------
      tokenContext context;
      keywordTrie keywords;
      keywordToStatementMap keywordPeek;
      statementLoaderMap statementLoaders;
      nameToAttributeMap attributeMap;

      int lastPeek;
      int lastPeekPosition;
      bool checkSemicolon;

      blockStatement root;
      attributePtrVector attributes;

      bool success;
      //================================

      parser_t();
      ~parser_t();

      //---[ Setup ]--------------------
      void clear();

      void parseSource(const std::string &source);
      void parseFile(const std::string &filename);

      void setSource(const std::string &source,
                     const bool isFile);
      void loadTokens();
      void parseTokens();

      keyword_t* getKeyword(token_t *token);
      opType_t getOperatorType(token_t *token);
      //================================

      //---[ Customization ]------------
      void addAttribute(attribute_t *attr);
      //================================

      //---[ Peek ]---------------------
      int peek();
      int uncachedPeek();

      void setupPeek();

      void skipNewlines();

      void loadAttributes(attributePtrVector &attrs);
      void loadAttribute(attributePtrVector &attrs);
      void addAttributesTo(attributePtrVector &attrs,
                           statement_t *smnt);

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

      void setArguments(variableVector &args);
      void getArgumentRanges(tokenRangeVector &argRanges);
      variable_t getArgument();

      class_t loadClassType();
      struct_t loadStructType();
      enum_t loadEnumType();
      union_t loadUnionType();
      //================================

      //---[ Loader Helpers ]-----------
      bool isEmpty();
      statement_t* getNextStatement(blockStatement &blockSmnt);
      //================================

      //---[ Statement Loaders ]--------
      void loadAllStatements(blockStatement &blockSmnt);

      statement_t* loadBlockStatement(blockStatement &up);

      statement_t* loadEmptyStatement(blockStatement &up);

      statement_t* loadExpressionStatement(blockStatement &up);

      statement_t* loadDeclarationStatement(blockStatement &up);

      statement_t* loadNamespaceStatement(blockStatement &up);

      statement_t* loadTypeDeclStatement(blockStatement &up);

      statement_t* loadFunctionStatement(blockStatement &up);

      void checkIfConditionStatementExists();
      void loadConditionStatements(blockStatement &up,
                                   statementPtrVector &statements,
                                   const int expectedCount);
      statement_t* loadConditionStatement(blockStatement &up);

      statement_t* loadIfStatement(blockStatement &up);
      statement_t* loadElifStatement(blockStatement &up);
      statement_t* loadElseStatement(blockStatement &up);

      statement_t* loadForStatement(blockStatement &up);
      statement_t* loadWhileStatement(blockStatement &up);
      statement_t* loadDoWhileStatement(blockStatement &up);

      statement_t* loadSwitchStatement(blockStatement &up);
      statement_t* loadCaseStatement(blockStatement &up);
      statement_t* loadDefaultStatement(blockStatement &up);
      statement_t* loadContinueStatement(blockStatement &up);
      statement_t* loadBreakStatement(blockStatement &up);

      statement_t* loadReturnStatement(blockStatement &up);

      statement_t* loadClassAccessStatement(blockStatement &up);

      statement_t* loadPragmaStatement(blockStatement &up);

      statement_t* loadGotoStatement(blockStatement &up);
      statement_t* loadGotoLabelStatement(blockStatement &up);
      //================================
    };
  }
}
#endif
