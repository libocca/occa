#include <occa/internal/lang/modes/openmp.hpp>
#include <occa/internal/lang/expr/expr.hpp>
#include <occa/internal/lang/builtins/attributes/atomic.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      openmpParser::openmpParser(const occa::json &settings_) :
        serialParser(settings_) {}

      void openmpParser::afterParsing() {
        serialParser::afterParsing();

        if (!success) return;
        setupOmpPragmas();

        if (!success) return;
        setupAtomics();
      }

      void openmpParser::setupOmpPragmas() {
        statementArray outerSmnts = (
          statementArray::from(root)
          .flatFilter([&](statement_t *smnt, const statementArray &path) {
              // Needs to be a @outer for-loop
              if (!isOuterForLoop(smnt)) {
                return false;
              }

              // Cannot have a parent @outer for-loop
              for (auto pathSmnt : path) {
                if (isOuterForLoop(pathSmnt)) {
                  return false;
                }
              }

              return true;
            })
        );

        const int count = (int) outerSmnts.length();
        for (int i = 0; i < count; ++i) {
          statement_t &outerSmnt = *(outerSmnts[i]);
          statement_t *parent = outerSmnt.up;
          if (!parent
              || !parent->is<blockStatement>()) {
            success = false;
            outerSmnt.printError("Unable to add [#pragma omp]");
            return;
          }
          // Add OpenMP Pragma
          blockStatement &outerBlock  = (blockStatement&) outerSmnt;
          blockStatement &parentBlock = *((blockStatement*) parent);
          pragmaStatement *pragmaSmnt = (
            new pragmaStatement((blockStatement*) parent,
                                pragmaToken(outerBlock.source->origin,
                                            "omp parallel for"))
          );
          parentBlock.addBefore(outerSmnt,
                                *pragmaSmnt);
        }
      }

      bool openmpParser::isOuterForLoop(statement_t *smnt) {
        return (
          (smnt->type() & statementType::for_)
          && smnt->hasAttribute("outer")
        );
      }

      void openmpParser::setupAtomics() {
        success &= attributes::atomic::applyCodeTransformation(
          root,
          transformBlockStatement,
          transformBasicExpressionStatement
        );
      }

      bool openmpParser::transformBlockStatement(blockStatement &blockSmnt) {
        blockStatement &parent = *(blockSmnt.up);

        pragmaStatement &atomicPragmaSmnt = *(
          new pragmaStatement(&parent,
                              pragmaToken(blockSmnt.source->origin,
                                          "omp critical"))
        );

        parent.addBefore(blockSmnt, atomicPragmaSmnt);

        return true;
      }

      bool openmpParser::transformBasicExpressionStatement(expressionStatement &exprSmnt) {
        blockStatement &parent = *(exprSmnt.up);

        pragmaStatement &atomicPragmaSmnt = *(
          new pragmaStatement(&parent,
                              pragmaToken(exprSmnt.source->origin,
                                          "omp atomic"))
        );

        parent.addBefore(exprSmnt, atomicPragmaSmnt);

        return true;
      }
    }
  }
}
