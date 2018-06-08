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
#include <occa/lang/modes/openmp.hpp>
#include <occa/lang/builtins/transforms/finders.hpp>

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
