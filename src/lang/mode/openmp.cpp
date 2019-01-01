#include <occa/lang/mode/openmp.hpp>
#include <occa/lang/transforms/builtins/finders.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      openmpParser::openmpParser(const occa::properties &settings_) :
        serialParser(settings_) {}

      void openmpParser::afterParsing() {
        serialParser::afterParsing();

        statementPtrVector outerSmnts;
        findOuterMostLoops(outerSmnts);

        const int count = (int) outerSmnts.size();
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

      void openmpParser::findOuterMostLoops(statementPtrVector &outerMostSmnts) {
        statementPtrVector outerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             root,
                             outerSmnts);

        const int count = (int) outerSmnts.size();
        for (int i = 0; i < count; ++i) {
          statement_t *outerSmnt = outerSmnts[i];
          statement_t *smnt = outerSmnt->up;
          while (smnt) {
            if (smnt->hasAttribute("outer")) {
              break;
            }
            smnt = smnt->up;
          }
          if (!smnt) {
            outerMostSmnts.push_back(outerSmnt);
          }
        }
      }
    }
  }
}
