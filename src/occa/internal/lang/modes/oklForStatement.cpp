#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/modes/okl.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      oklForStatement::oklForStatement(forStatement &forSmnt_,
                                       const std::string &source_,
                                       const bool printErrors_) :
        forSmnt(forSmnt_),
        source(source_),
        printErrors(printErrors_),
        iterator(NULL),
        initValue(NULL),
        checkOp(NULL),
        checkValue(NULL),
        updateOp(NULL),
        updateValue(NULL),
        valid(false) {

        const bool hasInner = forSmnt.hasAttribute("inner");
        const bool hasOuter = forSmnt.hasAttribute("outer");

        if (hasInner && hasOuter) {
          if (printErrors) {
            forSmnt.printError(sourceStr() + "Cannot have @inner and @outer");
          }
        } else if (!hasInner && !hasOuter) {
          if (printErrors) {
            forSmnt.printError(sourceStr() + "Missing @inner or @outer");
          }
        } else if (hasInner) {
          oklAttr = "inner";
        } else if (hasOuter) {
          oklAttr = "outer";
        }

        valid = (
          hasValidInit()
          && hasValidCheck()
          && hasValidUpdate()
        );
      }

      bool oklForStatement::isValid() {
        return valid;
      }

      bool oklForStatement::isValid(forStatement &forSmnt_,
                                    const std::string &source_,
                                    const bool printErrors_) {
        return oklForStatement(
          forSmnt_,
          source_,
          printErrors_
        ).isValid();
      }

      bool oklForStatement::hasValidInit() {
        statement_t &initSmnt = *(forSmnt.init);
        // Check for declaration
        if (initSmnt.type() != statementType::declaration) {
          if (printErrors) {
            initSmnt.printError(sourceStr() + "Expected a declaration statement");
          }
          return false;
        }
        // Can only have one declaration
        declarationStatement &declSmnt = (declarationStatement&) initSmnt;
        if (declSmnt.declarations.size() > 1) {
          if (printErrors) {
            declSmnt.declarations[1].printError(
              sourceStr() + "Can only have 1 iterator variable"
            );
          }
          return false;
        }
        // Get iterator and value
        variableDeclaration &decl = declSmnt.declarations[0];
        iterator  = &decl.variable();
        initValue = decl.value;
        // Valid types: {char, short, int, long}
        const type_t *type = iterator->vartype.flatten().type;
        if (!type ||
            ((*type != char_)  &&
             (*type != short_) &&
             (*type != int_))) {
          if (printErrors) {
            iterator->printError(sourceStr() + "Iterator variable needs to be of type"
                                 " [char, short, int, long]");
          }
          return false;
        }
        return true;
      }

      bool oklForStatement::hasValidCheck() {
        statement_t &checkSmnt = *(forSmnt.check);
        // Check an expression statement exists
        if (checkSmnt.type() != statementType::expression) {
          if (printErrors) {
            checkSmnt.printError(sourceStr() + "Expected comparing ["
                                 + iterator->name()
                                 + "] with some bound");
          }
          return false;
        }
        // Check valid operator (<, <=, >=, >)
        exprNode &expr = *(((expressionStatement&) checkSmnt).expr);
        if (expr.type() != exprNodeType::binary) {
          if (printErrors) {
            checkSmnt.printError(sourceStr() + "{0} Expected to compare ["
                                 + iterator->name()
                                 + "] with one of these operators [<, <=, >=, >]");
          }
          return false;
        }
        // Set check operator
        checkOp = (binaryOpNode*) &expr;
        opType_t checkOpType = checkOp->opType();
        if (!(checkOpType & (operatorType::lessThan      |
                             operatorType::lessThanEq    |
                             operatorType::greaterThanEq |
                             operatorType::greaterThan))) {
          if (printErrors) {
            checkSmnt.printError(sourceStr() + "{1} Expected to compare ["
                                 + iterator->name()
                                 + "] with one of these operators [<, <=, >=, >]");
          }
          return false;
        }
        checkIsInclusive = (
          checkOpType & (operatorType::lessThanEq    |
                         operatorType::greaterThanEq)
        );
        // Set check value
        int checkOrder = usesIterator(*checkOp,
                                      checkValue);
        if (!checkOrder) {
          if (printErrors) {
            checkSmnt.printError(sourceStr() + "{2} Expected to compare ["
                                 + iterator->name()
                                 + "] with one of these operators [<, <=, >=, >]");
          }
          return false;
        }
        checkValueOnRight = (checkOrder < 0);
        return true;
      }

      bool oklForStatement::hasValidUpdate() {
        statement_t &updateSmnt = *(forSmnt.update);
        // Check an expression statement exists
        if (updateSmnt.type() != statementType::expression) {
          if (printErrors) {
            updateSmnt.printError(sourceStr() + "Expected to update ["
                                  + iterator->name()
                                  + "]");
          }
          return false;
        }
        // Check valid operator (++, --, +=, -=)
        exprNode *updateExpr = ((expressionStatement&) updateSmnt).expr;
        udim_t eType = updateExpr->type();
        if (!(eType & (exprNodeType::leftUnary  |
                       exprNodeType::rightUnary |
                       exprNodeType::binary))) {
          if (printErrors) {
            updateSmnt.printError(sourceStr() + "Expected update ["
                                  + iterator->name()
                                  + "] with one of these operators [++, --, +=, -=]");
          }
          return false;
        }
        // Make sure we're using the same iterator variable
        bool validOp  = false;
        bool validVar = false;
        updateOp = (exprOpNode*) updateExpr;
        if (eType == exprNodeType::leftUnary) {
          leftUnaryOpNode &opNode = (leftUnaryOpNode&) *updateOp;
          opType_t opType = opNode.opType();
          validOp = (opType & (operatorType::leftIncrement |
                               operatorType::leftDecrement));
          validVar = usesIterator(opNode);
          positiveUpdate = (opType & operatorType::leftIncrement);
        }
        else if (eType == exprNodeType::rightUnary) {
          rightUnaryOpNode &opNode = (rightUnaryOpNode&) *updateOp;
          opType_t opType = opNode.opType();
          validOp = (opType & (operatorType::rightIncrement |
                               operatorType::rightDecrement));
          validVar = usesIterator(opNode);
          positiveUpdate = (opType & operatorType::rightIncrement);
        }
        else { // eType == exprNodeType::binary
          binaryOpNode &opNode = (binaryOpNode&) *updateOp;
          opType_t opType = opNode.opType();
          validOp = (opType & (operatorType::addEq |
                               operatorType::subEq));
          validVar = usesIterator(opNode, updateValue);
          positiveUpdate = (opType & operatorType::addEq);
        }
        if (!validOp) {
          if (printErrors) {
            updateOp->printError(sourceStr() + "Expected update ["
                                 + iterator->name()
                                 + "] with one of these operators [++, --, +=, -=]");
          }
          return false;
        }
        if (!validVar) {
          if (printErrors) {
            updateOp->startNode()->printError(sourceStr() + "Expected update ["
                                              + iterator->name()
                                              + "] with one of these operators [++, --, +=, -=]");
          }
          return false;
        }
        return true;
      }

      bool oklForStatement::usesIterator(leftUnaryOpNode &opNode) {
        if (opNode.value->type() != exprNodeType::variable) {
          return false;
        }
        variable_t &var = ((variableNode*) opNode.value)->value;
        return (&var == iterator);
      }

      bool oklForStatement::usesIterator(rightUnaryOpNode &opNode) {
        if (opNode.value->type() != exprNodeType::variable) {
          return false;
        }
        variable_t &var = ((variableNode*) opNode.value)->value;
        return (&var == iterator);
      }

      int oklForStatement::usesIterator(binaryOpNode &opNode,
                                        exprNode *&value) {

        if (opNode.leftValue->type() == exprNodeType::variable) {
          variable_t &var = ((variableNode*) opNode.leftValue)->value;
          if (&var == iterator) {
            value = opNode.rightValue;
            return -1;
          }
        }
        if (opNode.rightValue->type() == exprNodeType::variable) {
          variable_t &var = ((variableNode*) opNode.rightValue)->value;
          if (&var == iterator) {
            value = opNode.leftValue;
            return 1;
          }
        }
        return 0;
      }

      exprNode* oklForStatement::getIterationCount() {
        if (!valid) {
          return NULL;
        }

        exprNode *initInParen = initValue->wrapInParentheses();
        exprNode *count = (
          new binaryOpNode(iterator->source,
                           positiveUpdate ? op::sub : op::add,
                           *checkValue,
                           *initInParen)
        );
        delete initInParen;

        if (checkIsInclusive) {
          primitiveNode inc(iterator->source, 1);

          exprNode *countWithInc = (
            new binaryOpNode(iterator->source,
                             positiveUpdate ? op::sub : op::add,
                             *count,
                             inc)
          );
          delete count;
          count = countWithInc;
        }

        if (updateValue) {
          exprNode *updateInParen = updateValue->wrapInParentheses();

          primitiveNode one(iterator->source, 1);
          binaryOpNode boundCheck(iterator->source,
                                  positiveUpdate ? op::add : op::sub,
                                  *count,
                                  *updateInParen);
          binaryOpNode boundCheck2(iterator->source,
                                   positiveUpdate ? op::sub : op::add,
                                   boundCheck,
                                   one);
          exprNode *boundCheckInParen = boundCheck2.wrapInParentheses();

          exprNode *countWithUpdate = (
            new binaryOpNode(iterator->source,
                             op::div,
                             *boundCheckInParen,
                             *updateInParen)
          );
          delete count;
          delete updateInParen;
          delete boundCheckInParen;
          count = countWithUpdate;
        }

        return count;
      }

      exprNode* oklForStatement::makeDeclarationValue(exprNode &magicIterator) {
        if (!valid) {
          return NULL;
        }

        exprNode *blockValue = magicIterator.wrapInParentheses();
        if (updateValue) {
          exprNode *updateInParen = updateValue->wrapInParentheses();
          binaryOpNode mult(iterator->source,
                            op::mult,
                            *updateInParen,
                            *blockValue);
          delete updateInParen;
          delete blockValue;
          blockValue = mult.wrapInParentheses();
        }

        exprNode *initInParen = initValue->wrapInParentheses();
        binaryOpNode *value = (
          new binaryOpNode(iterator->source,
                           positiveUpdate ? op::add : op::sub,
                           *initInParen,
                           *blockValue)
        );

        delete blockValue;
        delete initInParen;

        return value;
      }

      bool oklForStatement::isInnerLoop() {
        return oklAttr == "inner";
      }

      bool oklForStatement::isOuterLoop() {
        return oklAttr == "outer";
      }

      int oklForStatement::oklLoopIndex() {
        return getOklLoopIndex(forSmnt, oklAttr);
      }

      int oklForStatement::getOklLoopIndex(forStatement &forSmnt_, const std::string &oklAttr_) {
        attributeToken_t &oklAttrToken = forSmnt_.attributes[oklAttr_];
        if (oklAttrToken.args.size()) {
          // The attribute is either @outer(N) or @inner(N)
          // We need to evaluate the value inside the attribute first though
          return (int) oklAttrToken.args[0].expr->evaluate();
        }

        // Find index by looking at how many for-loops of the same attribute are inside
        int maxInnerCount = 0;
        forOklForLoopStatements(
          forSmnt_,
          [&](forStatement &innerForSmnt,
              const std::string innerAttr,
              const statementArray &path) {

            const auto isInnerOklLoop = [&](statement_t *smnt) {
              return (
                (smnt != &forSmnt_)
                && (smnt->type() & statementType::for_)
                && smnt->hasAttribute(oklAttr_)
              );
            };

            // Check innerForSmnt first
            // Then add to the matched statements between forSmnt_ and innerForSmnt
            const int innerCount = (
              isInnerOklLoop(&innerForSmnt)
              + path.filter(isInnerOklLoop).length()
            );

            if (innerCount > maxInnerCount) {
              maxInnerCount = innerCount;
            }
          });

        return maxInnerCount;
      }

      statementArray oklForStatement::getOklLoopPath() {
        return getOklLoopPath(forSmnt);
      }

      statementArray oklForStatement::getOklLoopPath(forStatement &forSmnt_) {
        statementArray oklParentPath = (
          forSmnt_
          .getParentPath()
          .filter([&](statement_t *smnt) {
              return (
                (smnt->type() & statementType::for_)
                && (
                  smnt->hasAttribute("inner")
                  || smnt->hasAttribute("outer")
                ));
            })
        );

        // forSmnt_ is not in the getParentPath() list
        // so we need to push it manually
        oklParentPath.push(&forSmnt_);

        return oklParentPath;
      }

      std::string oklForStatement::sourceStr() {
        if (source.size()) {
          return ("[" + source + "] ");
        }
        return "";
      }

      void oklForStatement::printWarning(const std::string &message) {
        forSmnt.printWarning(message);
      }

      void oklForStatement::printError(const std::string &message) {
        forSmnt.printError(message);
      }
    }
  }
}
