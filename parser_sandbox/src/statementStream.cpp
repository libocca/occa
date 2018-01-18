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
      tokens(root),
      parentStatement(NULL) {}

    statementStream::statementStream(file_t *file_,
                                     const char *root) :
      tokens(file_, root),
      parentStatement(NULL) {}

    void statementStream::preprint(std::ostream &out) {
      tokens.preprint(out);
    }

    void statementStream::postprint(std::ostream &out) {
      tokens.postprint(out);
    }

    void statementStream::clear() {
      tokenStack.clear();
      attributes.clear();
    }

    token_t* statementStream::getToken() {
      tokens.push();
      token_t *token = tokens.getToken();
      if ((token == NULL)
          && !tokens.isEmpty()) {
        tokens.popAndRewind();
        handleTokenError();
      }
      tokens.pop();
      tokenStack.push_back(token);
      return token;
    }

    int statementStream::peek() {
      token_t *token = getToken();
      const int tokenType = (token
                             ? token->type()
                             : tokenType::none);

      if (tokenType & tokenType::identifier) {
        return peekForIdentifier();
      }
      if (tokenType & tokenType::op) {
        return peekForOperator();
      }
      if (tokenType & (tokenType::primitive |
                       tokenType::char_     |
                       tokenType::string)) {
        return statementType::expression;
      }
      if (tokenType & tokenType::attribute) {
        // attributes.push_back(&(token->to<attributeToken>()));
        tokenStack.pop_back();
        return peek();
      }
      return statementType::none;
    }

    int statementStream::peekForIdentifier() {
      return statementType::none;
    }

    int statementStream::peekForOperator() {
      return statementType::none;
    }

    statement_t* statementStream::getStatement() {
      statement_t *statement;
      switch (peek()) {
      case statementType::directive:
        statement = getDirectiveStatement();   break;
      case statementType::block:
        statement = getBlockStatement();       break;
      case statementType::typeDecl:
        statement = getTypeDeclStatement();    break;
      case statementType::classAccess:
        statement = getClassAccessStatement(); break;
      case statementType::expression:
        statement = getExpressionStatement();  break;
      case statementType::declaration:
        statement = getDeclarationStatement(); break;
      case statementType::goto_:
        statement = getGotoStatement();        break;
      case statementType::gotoLabel:
        statement = getGotoLabelStatement();   break;
      case statementType::namespace_:
        statement = getNamespaceStatement();   break;
      case statementType::while_:
        statement = getWhileStatement();       break;
      case statementType::for_:
        statement = getForStatement();         break;
      case statementType::switch_:
        statement = getSwitchStatement();      break;
      case statementType::case_:
        statement = getCaseStatement();        break;
      case statementType::continue_:
        statement = getContinueStatement();    break;
      case statementType::break_:
        statement = getBreakStatement();       break;
      case statementType::return_:
        statement = getReturnStatement();
      }
      if ((statement == NULL) &&
          !tokens.isEmpty()) {
        printError("Not able to create statement for:");
        return NULL;
      }
      const int attributeCount = (int) attributes.size();
      for (int i = 0; i < attributeCount; ++i) {
        statement->addAttribute(attributes[i]);
      }
      return statement;
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

    void statementStream::handleTokenError() {
      if (parentStatement == NULL) {
        tokens.skipTo('\n');
      }
    }
  }
}
