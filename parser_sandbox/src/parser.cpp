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

      attributes.clear();

      success = false;
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
      context.resetPosition();
    }

    void parser_t::parse() {
      loadTokens();

      root = new blockStatement();
      up   = root;
      loadBlockStatement(*root);
    }

    // void parser_t::findPairs(intVector &pairs) {
      // const int tokens = (int) context.tokens.size();
      // for (int i = 0; i < tokens; ++i) {
      //   token_t *token = context.tokens[i];
      //   if (!(token->type() & tokenType::op)) {
      //     continue;
      //   }
      //   operatorToken &opToken = token->to<operatorToken>();
      //   if (!(opToken->opType() & pair)) {
      //     continue;
      //   }

      //   operatorToken &errorToken = *(pairs.back());
      //   pairs.
      // }
    // }

    void parser_t::loadBlockStatement(blockStatement &smnt) {
    }

    void parser_t::loadForStatement(forStatement &smnt) {
    }

    void parser_t::loadWhileStatement(whileStatement &smnt) {
    }

    void parser_t::loadIfStatement(ifStatement &smnt) {
    }

    void parser_t::loadElseIfStatement(elifStatement &smnt) {
    }

    void parser_t::loadElseStatement(elseStatement &smnt) {
    }

    void* parser_t::getDeclaration() {
      return NULL;
    }

    void* parser_t::getExpression() {
      return NULL;
    }

    void* parser_t::getFunction() {
      return NULL;
    }

    void* parser_t::getAttribute() {
      return NULL;
    }
  }
}
