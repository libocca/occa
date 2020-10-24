#include <occa/lang/expr.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/attributes/atomic.hpp>

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

      void atomic::applyCodeTransformation(
        blockStatement &root,
        blockSmntVoidCallback transformBlockSmnt,
        exprSmntVoidCallback transformBasicExprSmnt
      ) {
        root.children
            .flatFilterByStatementType(
              statementType::expression | statementType::block,
              "atomic"
            )
            .forEach([&](statement_t *smnt) {
                if (smnt->type() & statementType::expression) {
                  expressionStatement &exprSmnt = (expressionStatement&) *smnt;
                  applyExpressionCodeTransformation(
                    exprSmnt,
                    transformBlockSmnt,
                    transformBasicExprSmnt
                  );
                } else {
                  blockStatement &blockSmnt = (blockStatement&) *smnt;
                  applyBlockCodeTransformation(
                    blockSmnt,
                    transformBlockSmnt,
                    transformBasicExprSmnt
                  );
                }
              });
      }

      void atomic::applyExpressionCodeTransformation(
        expressionStatement &exprSmnt,
        blockSmntVoidCallback transformBlockSmnt,
        exprSmntVoidCallback transformBasicExprSmnt
      ) {
        blockStatement &parent = *(exprSmnt.up);

        if (attributes::atomic::isBasicExpression(exprSmnt)) {
          transformBasicExprSmnt(exprSmnt);
          return;
        }

        // Create a block statement and make a critical region around it
        blockStatement &blockSmnt = *(
          new blockStatement(&parent, exprSmnt.source)
        );
        exprSmnt.replaceWith(blockSmnt);
        blockSmnt.add(exprSmnt);

        transformBlockSmnt(blockSmnt);
      }

      void atomic::applyBlockCodeTransformation(
        blockStatement &blockSmnt,
        blockSmntVoidCallback transformBlockSmnt,
        exprSmntVoidCallback transformBasicExprSmnt
      ) {
        if (blockSmnt.size() == 1
            && (blockSmnt[0]->type() & statementType::expression)) {
          expressionStatement &exprSmnt = (expressionStatement&) *blockSmnt[0];
          if (attributes::atomic::isBasicExpression(exprSmnt)) {
            // Remove unneeded block statement
            blockSmnt.remove(exprSmnt);
            blockSmnt.replaceWith(exprSmnt);
            delete &blockSmnt;

            transformBasicExprSmnt(exprSmnt);
            return;
          }
        }

        transformBlockSmnt(blockSmnt);
      }

      bool atomic::isBasicExpression(expressionStatement &exprSmnt) {
        const opType_t &opType = expr(exprSmnt.expr).opType();
        return opType & (
          operatorType::assignment
          | operatorType::increment
          | operatorType::decrement
        );
      }
    }
  }
}
