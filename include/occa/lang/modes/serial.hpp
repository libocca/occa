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
#ifndef OCCA_PARSER_MODES_SERIAL_HEADER
#define OCCA_PARSER_MODES_SERIAL_HEADER

#include <occa/lang/parser.hpp>
#include <occa/lang/builtins/transforms/finders.hpp>

namespace occa {
  namespace lang {
    namespace transforms {
      class smntTreeNode;
    }

    namespace okl {
      class serialParser : public parser_t {
      public:
        static const std::string exclusiveIndexName;

        serialParser(const occa::properties &settings_ = occa::properties());

        virtual void onClear();

        virtual void afterParsing();

        void setupHeaders();

        void setupKernels();

        static void setupKernel(functionDeclStatement &kernelSmnt);

        void setupExclusives();

        void setupExclusiveDeclarations(statementExprMap &exprMap);
        void setupExclusiveDeclaration(declarationStatement &declSmnt);
        bool exclusiveIsDeclared(declarationStatement &declSmnt);

        void setupExclusiveIndices();

        static bool exclusiveVariableMatcher(exprNode &expr);

        static bool exclusiveInnerLoopMatcher(statement_t &smnt);

        void getInnerMostLoops(transforms::smntTreeNode &innerRoot,
                               statementPtrVector &loopSmnts);


        static exprNode* updateExclusiveExprNodes(statement_t &smnt,
                                                  exprNode &expr,
                                                  const bool isBeingDeclared);
      };
    }
  }
}

#endif
