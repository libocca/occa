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
      root(),
      up(&root) {
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

      root.clear();
      up = &root;

      freeKeywords(keywords);
      attributes.clear();

      success = true;
    }

    void parser_t::parseSource(const std::string &source) {
      setSource(source, false);
      if (!success) {
        return;
      }
      parseTokens();
    }

    void parser_t::parseFile(const std::string &filename) {
      setSource(filename, true);
      if (!success) {
        return;
      }
      parseTokens();
    }

    void parser_t::setSource(const std::string &source,
                             const bool isFile) {
      clear();

      if (isFile) {
        tokenizer.set(new file_t(source));
      } else {
        tokenizer.set(source.c_str());
      }

      loadTokens();
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
      success = !context.hasError;

      if (success) {
        getKeywords(keywords);
      }
    }

    void parser_t::parseTokens() {
      loadChildStatements(root);
    }

    keyword_t* parser_t::getKeyword(token_t *token) {
      if (!(token_t::safeType(token) & tokenType::identifier)) {
        return NULL;
      }

      identifierToken &identifier = token->to<identifierToken>();
      return keywords.get(identifier.value).value();
    }

    int parser_t::peek() {
      const int tokens = context.size();
      if (!tokens) {
        return statementType::empty;
      }

      int stype = statementType::empty;
      int tokenIndex = 0;

      while (success                        &&
             (stype & statementType::empty) &&
             (tokenIndex < tokens)) {

        token_t *token = context[tokenIndex];
        const int tokenType = token->type();

        if (tokenType & tokenType::identifier) {
          return peekIdentifier(tokenIndex);
        }

        if (tokenType & tokenType::op) {
          return peekOperator(tokenIndex);
        }

        if (tokenType & (tokenType::primitive |
                         tokenType::string    |
                         tokenType::char_)) {
          return statementType::expression;
        }

        if (tokenType & tokenType::pragma) {
          return statementType::pragma;
        }

        ++tokenIndex;
      }

      return (success
              ? stype
              : statementType::none);
    }

    int parser_t::peekIdentifier(const int tokenIndex) {
      token_t *token     = context[tokenIndex];
      keyword_t *keyword = getKeyword(token);
      // Test for : for it to be a goto label
      // const int gotoLabel   = (1 << 9);
      if (!keyword) {
        if (isGotoLabel(tokenIndex + 1)) {
          return statementType::gotoLabel;
        }
        // TODO: Attempt to find error by guessing the keyword type
        token->printError("Unknown identifier");
        success = false;
        return statementType::none;
      }

      const int kType = keyword->type();

      if (kType & (keywordType::qualifier |
                   keywordType::type)) {
        return statementType::declaration;
      }

      if (kType & keywordType::if_) {
        return statementType::if_;
      }

      if (kType & keywordType::else_) {
        keyword_t *nextKeyword = getKeyword(context[tokenIndex + 1]);
        if (nextKeyword &&
            (nextKeyword->type() & keywordType::if_)) {
          return statementType::elif_;
        }
        return statementType::else_;
      }

      if (kType & keywordType::switch_) {
        return statementType::switch_;
      }

      if (kType & keywordType::case_) {
        return statementType::case_;
      }

      if (kType & keywordType::default_) {
        return statementType::default_;
      }

      if (kType & keywordType::for_) {
        return statementType::for_;
      }

      if (kType & (keywordType::while_ |
                   keywordType::do_)) {
        return statementType::while_;
      }

      if (kType & keywordType::break_) {
        return statementType::break_;
      }

      if (kType & keywordType::continue_) {
        return statementType::continue_;
      }

      if (kType & keywordType::return_) {
        return statementType::return_;
      }

      if (kType & keywordType::classAccess) {
        return statementType::classAccess;
      }

      if (kType & keywordType::namespace_) {
        return statementType::namespace_;
      }

      if (kType & keywordType::goto_) {
        return statementType::goto_;
      }

      return statementType::expression;
    }

    bool parser_t::isGotoLabel(const int tokenIndex) {
      token_t *token = context[tokenIndex];
      if (!(token_t::safeType(token) & tokenType::op)) {
        return false;
      }
      operatorToken &opToken = token->to<operatorToken>();
      return (opToken.getOpType() & operatorType::colon);
    }

    int parser_t::peekOperator(const int tokenIndex) {
      const opType_t opType = (context[tokenIndex]
                               ->to<operatorToken>()
                               .getOpType());

      if (opType & operatorType::braceStart) {
        return statementType::block;
      }
      if (opType & operatorType::attribute) {
        return statementType::attribute;
      }
      return statementType::expression;
    }

    void parser_t::loadChildStatements(blockStatement &smnt) {
    }
  }
}
