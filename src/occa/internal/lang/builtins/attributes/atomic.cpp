#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/attributes/atomic.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      atomic::atomic() {}

      const std::string& atomic::name() const {
        static std::string name_ = "atomic";
        return name_;
      }

      bool atomic::forStatementType(const int sType) const {
        return (
          sType & (
            statementType::expression
            | statementType::block
          )
        );
      }

      bool atomic::isValid(const attributeToken_t &attr) const {
        if (attr.args.size() || attr.kwargs.size()) {
          attr.printError("[@atomic] does not take arguments");
          return false;
        }
        return true;
      }

      bool atomic::applyCodeTransformation(
        blockStatement &root,
        blockSmntBoolCallback transformBlockSmnt,
        exprSmntBoolCallback transformBasicExprSmnt
      ) {
        bool success = true;
        statementArray::from(root)
            .flatFilterByStatementType(
              statementType::expression | statementType::block,
              "atomic"
            )
            .forEach([&](statement_t *smnt) {
                if (smnt->type() & statementType::expression) {
                  expressionStatement &exprSmnt = (expressionStatement&) *smnt;
                  success &= applyExpressionCodeTransformation(
                    exprSmnt,
                    transformBlockSmnt,
                    transformBasicExprSmnt
                  );
                } else {
                  blockStatement &blockSmnt = (blockStatement&) *smnt;
                  success &= applyBlockCodeTransformation(
                    blockSmnt,
                    transformBlockSmnt,
                    transformBasicExprSmnt
                  );
                }
              });
        return success;
      }

      bool atomic::applyExpressionCodeTransformation(
        expressionStatement &exprSmnt,
        blockSmntBoolCallback transformBlockSmnt,
        exprSmntBoolCallback transformBasicExprSmnt
      ) {
        blockStatement &parent = *(exprSmnt.up);

        if (attributes::atomic::isBasicExpression(exprSmnt)) {
          return transformBasicExprSmnt(exprSmnt);
        }

        // Create a block statement and make a critical region around it
        blockStatement &blockSmnt = *(
          new blockStatement(&parent, exprSmnt.source)
        );
        exprSmnt.replaceWith(blockSmnt);
        blockSmnt.add(exprSmnt);

        return transformBlockSmnt(blockSmnt);
      }

      bool atomic::applyBlockCodeTransformation(
        blockStatement &blockSmnt,
        blockSmntBoolCallback transformBlockSmnt,
        exprSmntBoolCallback transformBasicExprSmnt
      ) {
        if (blockSmnt.size() == 1
            && (blockSmnt[0]->type() & statementType::expression)) {
          expressionStatement &exprSmnt = (expressionStatement&) *blockSmnt[0];
          if (attributes::atomic::isBasicExpression(exprSmnt)) {
            // Remove unneeded block statement
            blockSmnt.remove(exprSmnt);
            blockSmnt.replaceWith(exprSmnt);
            delete &blockSmnt;

            return transformBasicExprSmnt(exprSmnt);
          }
        }

        return transformBlockSmnt(blockSmnt);
      }

      bool atomic::isBasicExpression(expressionStatement &exprSmnt) {
        const opType_t &opType = expr(exprSmnt.expr).opType();
        return opType & (
          operatorType::addEq
          | operatorType::subEq
          | operatorType::increment
          | operatorType::decrement
        );
      }
    }
  }
}
