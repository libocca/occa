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
#include "parser.hpp"

namespace occa {
  namespace lang {
    parser_t::parser_t() :
      unknownFilter(true),
      root(NULL),
      up(NULL) {
      // Properly implement `identifier-nondigit` for identifiers
      // Meanwhile, we use the unknownFilter
      stream = (tokenizer
                .filter(unknownFilter)
                .map(preprocessor)
                .map(stringMerger)
                .map(newlineMerger));
    }

    parser_t::~parser_t() {
      clear();
    }

    void parser_t::clear() {
      tokenizer.clear();
      preprocessor.clear();
      context.clear();

      delete root;
      root = NULL;
      up   = NULL;

      freeKeywords(keywords);
      attributes.clear();

      success = true;
    }

    void parser_t::parseSource(const std::string &source) {
      clear();
      tokenizer.set(source.c_str());
      parse();
    }

    void parser_t::parseFile(const std::string &filename) {
      clear();
      tokenizer.set(new file_t(filename));
      parse();
    }

    void parser_t::loadTokens() {
      token_t *token;
      while (!stream.isEmpty()) {
        stream >> token;
        context.tokens.push_back(token);
      }

      if (tokenizer.errors ||
          preprocessor.errors) {
        success = false;
        return;
      }

      context.setup();
      if (context.hasError) {
        success = false;
      }
    }

    void parser_t::parse() {
      loadTokens();
      if (!success) {
        return;
      }

      getKeywords(keywords);

      root = new blockStatement();
      up   = root;
      loadChildStatements(*root);
    }

    keyword_t* parser_t::getKeyword(token_t *token) {
      if (!(token->type() & tokenType::identifier)) {
        return NULL;
      }

      identifierToken &identifier = token->to<identifierToken>();
      keywordTrie::result_t result = keywords.get(identifier.value);
      if (!result.success()) {
        identifier.printError("Unknown identifier");
        success = false;
        return NULL;
      }

      return result.value();
    }

    int parser_t::peek() {
      return 0;
#if 0
      const int tokens = context.size();
      if (!tokens) {
        return statementType::empty;
      }

      token_t *token = context[0];
      // keyword_t *keyword = getKeyword(token);
      if (!success) {
        return statementType::empty;
      }

      const int kType = keyword.type();
      if (kType & keywordType::) {
        return statementType::expression;
      }
      if (kType & keywordType::) {
        return statementType::expression;
      }
      if (kType & keywordType::) {
        return statementType::expression;
      }
      if (kType & keywordType::) {
        return statementType::expression;
      }
      if (kType & keywordType::) {
        return statementType::expression;
      }

      const int empty       = (1 << 1);
      const int pragma      = (1 << 2);
      const int block       = (1 << 3);
      const int typeDecl    = (1 << 4);
      const int classAccess = (1 << 5);
      const int expression  = (1 << 6);
      const int declaration = (1 << 7);
      const int goto_       = (1 << 8);
      const int gotoLabel   = (1 << 9);
      const int namespace_  = (1 << 10);
      const int if_         = (1 << 11);
      const int elif_       = (1 << 12);
      const int else_       = (1 << 13);
      const int for_        = (1 << 14);
      const int while_      = (1 << 15);
      const int switch_     = (1 << 16);
      const int case_       = (1 << 17);
      const int continue_   = (1 << 18);
      const int break_      = (1 << 19);
      const int return_     = (1 << 20);
      const int attribute   = (1 << 21);
#endif
    }

    void parser_t::loadChildStatements(blockStatement &smnt) {
    }
  }
}
