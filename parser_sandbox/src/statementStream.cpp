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
#include "statementStream.hpp"

namespace occa {
  namespace lang {
    statementStream::statementStream(const char *root) :
      tokens(root) {
    }

    statementStream::statementStream(file_t *file_,
                                     const char *root) :
      tokens(file_, root) {
    }

    void statementStream::preprint(std::ostream &out) {
      tokens.preprint(out);
    }

    void statementStream::postprint(std::ostream &out) {
      tokens.postprint(out);
    }

    int statementStream::peek() {
      tokens.push();
      // token_t *token = tokens.getToken();
      tokens.pop();
      return 0;
    }

    statement_t* statementStream::getStatement() {
      return NULL;
    }

    statement_t* statementStream::getEmptyStatement() {
      return NULL;
    }

    statement_t* statementStream::getDirectiveStatement() {
      return NULL;
    }

    statement_t* statementStream::getBlockStatement() {
      return NULL;
    }

    statement_t* statementStream::getTypeDeclStatement() {
      return NULL;
    }

    statement_t* statementStream::getClassAccessStatement() {
      return NULL;
    }

    statement_t* statementStream::getExpressionStatement() {
      return NULL;
    }

    statement_t* statementStream::getDeclarationStatement() {
      return NULL;
    }

    statement_t* statementStream::getGotoStatement() {
      return NULL;
    }

    statement_t* statementStream::getGotoLabelStatement() {
      return NULL;
    }

    statement_t* statementStream::getNamespaceStatement() {
      return NULL;
    }

    statement_t* statementStream::getWhileStatement() {
      return NULL;
    }

    statement_t* statementStream::getForStatement() {
      return NULL;
    }

    statement_t* statementStream::getSwitchStatement() {
      return NULL;
    }

    statement_t* statementStream::getCaseStatement() {
      return NULL;
    }

    statement_t* statementStream::getContinueStatement() {
      return NULL;
    }

    statement_t* statementStream::getBreakStatement() {
      return NULL;
    }

    statement_t* statementStream::getReturnStatement() {
      return NULL;
    }
  }
}
