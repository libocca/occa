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
#include <occa/lang/attribute.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/statementTransform.hpp>
#include <occa/lang/parser.hpp>

namespace occa {
  namespace lang {
    statementTransform::statementTransform() :
      downToUp(true),
      validStatementTypes(statementType::none) {}

    bool statementTransform::apply(statement_t &smnt) {
      statement_t *smntPtr = &smnt;
      return transformStatementInPlace(smntPtr);
    }

    statement_t* statementTransform::transform(statement_t &smnt) {
      if (!(smnt.type() & validStatementTypes)) {
        return &smnt;
      }
      return transformStatement(smnt);
    }

    statement_t* statementTransform::transformBlockStatement(blockStatement &smnt) {
      if (downToUp) {
        if (!transformChildrenStatements(smnt)
            || !transformInnerStatements(smnt)) {
          return NULL;
        }
        return transform(smnt);
      }

      // Up to down
      statement_t* newSmnt = transform(smnt);
      if (!newSmnt) {
        return NULL;
      }

      if ((newSmnt != &smnt)
          && !newSmnt->is<blockStatement>()) {
        delete &smnt;
        return newSmnt;
      }
      blockStatement &newBlockSmnt = *((blockStatement*) newSmnt);

      if (!transformInnerStatements(newBlockSmnt)
          || !transformChildrenStatements(newBlockSmnt)) {
        if (newSmnt != &smnt) {
          delete newSmnt;
        }
        return NULL;
      }
      if (newSmnt != &smnt) {
        delete &smnt;
      }
      return newSmnt;
    }

    bool statementTransform::transformChildrenStatements(blockStatement &smnt) {
      // Transform children
      const int count = (int) smnt.children.size();
      for (int i = 0; i < count; ++i) {
        statement_t *&child = smnt.children[i];
        if (child
            && !transformStatementInPlace(child)) {
          return false;
        }
      }
      return true;
    }

    bool statementTransform::transformStatementInPlace(statement_t *&smnt) {
      if (!smnt) {
        return true;
      }
      statement_t *newSmnt = NULL;

      // Treat blockStatements differently
      if (smnt->is<blockStatement>()) {
        newSmnt = transformBlockStatement((blockStatement&) *smnt);
      } else {
        newSmnt = transform(*smnt);
      }
      // Statement wasn't replaced
      if (newSmnt == smnt) {
        return true;
      }
      // Error happened
      if (!newSmnt) {
        return false;
      }
      // Swap children
      newSmnt->up = smnt->up;
      delete smnt;
      smnt = newSmnt;
      return true;
    }

    bool statementTransform::transformInnerStatements(blockStatement &smnt) {
      const int sType = smnt.type();
      if (sType & statementType::for_) {
        return transformForInnerStatements(smnt.to<forStatement>());
      } else if (sType & statementType::if_) {
        return transformIfInnerStatements(smnt.to<ifStatement>());
      } else if (sType & statementType::elif_) {
        return transformElifInnerStatements(smnt.to<elifStatement>());
      } else if (sType & statementType::while_) {
        return transformWhileInnerStatements(smnt.to<whileStatement>());
      } else if (sType & statementType::switch_) {
        return transformSwitchInnerStatements(smnt.to<switchStatement>());
      }
      return true;
    }

    bool statementTransform::transformForInnerStatements(forStatement &smnt) {
      return (
        transformStatementInPlace(smnt.init)
        && transformStatementInPlace(smnt.check)
        && transformStatementInPlace(smnt.update)
      );
    }

    bool statementTransform::transformIfInnerStatements(ifStatement &smnt) {
      if (!transformStatementInPlace(smnt.condition)) {
        return false;
      }

      // Elif statements
      const int elifCount = (int) smnt.elifSmnts.size();
      int newCount = 0;
      for (int i = 0; i < elifCount; ++i) {
        statement_t *elifSmnt = smnt.elifSmnts[i];
        if (!transformStatementInPlace(elifSmnt)) {
          return false;
        }
        if (elifSmnt &&
            !elifSmnt->is<elifStatement>()) {
          printError("Elif transform became a non-elif statement");
          return false;
        }
        if (elifSmnt) {
          smnt.elifSmnts[newCount++] = (elifStatement*) elifSmnt;
        }
      }
      smnt.elifSmnts.resize(newCount);

      // Else statement
      statement_t *elseSmnt = smnt.elseSmnt;
      if (!elseSmnt) {
        return true;
      }
      if (!transformStatementInPlace(elseSmnt)) {
        return false;
      }
      if (elseSmnt &&
          !elseSmnt->is<elseStatement>()) {
        printError("Else transform became a non-else statement");
        return false;
      }
      smnt.elseSmnt = (elseStatement*) elseSmnt;
      return true;
    }

    bool statementTransform::transformElifInnerStatements(elifStatement &smnt) {
      return transformStatementInPlace(smnt.condition);
    }

    bool statementTransform::transformWhileInnerStatements(whileStatement &smnt) {
      return transformStatementInPlace(smnt.condition);
    }

    bool statementTransform::transformSwitchInnerStatements(switchStatement &smnt) {
      return transformStatementInPlace(smnt.condition);
    }
  }
}
