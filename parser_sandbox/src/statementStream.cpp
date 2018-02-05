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
#if 0
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
      token_t *token = tokens.getToken();
      if (token) {
        tokenStack.push_back(token);
      }
      return token;
    }

    void statementStream::popTokens(const int count) {
      const int total = (int) tokenStack.size();
      for (int i = 0; i < count; ++i) {
        token_t *token = tokenStack[total - i - 1];
        delete token;
        tokenStack.pop_back();
      }
    }

    void statementStream::popToken() {
      popTokens(1);
    }

    void statementStream::clearTokens() {
      popTokens((int) tokenStack.size());
    }

    int statementStream::peek() {
      token_t *token = getToken();
      const int tokenType = (token
                             ? token->type()
                             : tokenType::none);

      if (tokenType & tokenType::identifier) {
        return peekForIdentifier(token->to<identifierToken>());
      }
      if (tokenType & tokenType::op) {
        return peekForOperator(token->to<operatorToken>());
      }
      if (tokenType & (tokenType::primitive |
                       tokenType::char_     |
                       tokenType::string)) {
        return statementType::expression;
      }

      return statementType::none;
    }

    int statementStream::peekForIdentifier(identifierToken &token) {
      keywordTrie::result_t result = keywords.get(token.value);
      if (!result.success()) {

      }
      // type | qualifier -> statementType::typeDecl
      // goto             -> statementType::gotoLabel
      // variable         -> statementType::expression

      // "goto"           -> statementType::goto_
      // "namespace"      -> statementType::namespace_
      // "while"          -> statementType::while_
      // "for"            -> statementType::for_
      // "switch"         -> statementType::switch_
      // "case"           -> statementType::case_
      // "continue"       -> statementType::continue_
      // "break"          -> statementType::break_
      // "return"         -> statementType::return_
      // "typedef" | "class" | "struct" | "enum" | "union" | ?
      //                  -> statementType::declaration
      return statementType::none;
    }

    int statementStream::peekForOperator(operatorToken &token) {
      const opType_t opType = token.opType;
      if (opType & operatorType::hash) {
        return statementType::directive;
      }
      if (opType & operatorType::hashHash) {
        handleTokenError();
        return statementType::none;
      }
      if (opType & operatorType::braceStart) {
        return statementType::block;
      }
      if (opType & operatorType::attribute) {
        loadAttribute();
        return statementType::attribute;
      }
      return statementType::expression;
    }

    void statementStream::loadAttribute() {
      token_t *token = getToken();
      const int tokenType = (token
                             ? token->type()
                             : tokenType::none);
      if (tokenType & tokenType::identifier) {
        // TODO: Error at the @ token
        handleTokenError();
        return;
      }
      const std::string name = ptoken->to<identifierToken>().value;
      popTokens(2); // [@][name]

      // TODO: Finish parsing the () tokens

      attributes.push_back(name);
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
      attributes.clear();
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
#endif
