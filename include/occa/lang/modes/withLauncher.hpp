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
#ifndef OCCA_PARSER_MODES_WITHLAUNCHER_HEADER
#define OCCA_PARSER_MODES_WITHLAUNCHER_HEADER

#include <occa/lang/parser.hpp>
#include <occa/lang/modes/serial.hpp>
#include <occa/lang/builtins/transforms/finders.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class withLauncher : public parser_t {
      public:
        serialParser hostParser;

        withLauncher(const occa::properties &settings_ = occa::properties());

        //---[ Public ]-----------------
        virtual bool succeeded() const;

        void writeHostSourceToFile(const std::string &filename) const;
        //==============================

        void hostClear();

        void afterParsing();

        virtual void beforeKernelSplit();
        virtual void afterKernelSplit();

        void setOKLLoopIndices();

        void setupHostParser();

        void removeHostOuterLoops(functionDeclStatement &kernelSmnt);

        bool isOuterMostOuterLoop(forStatement &forSmnt);

        bool isOuterMostInnerLoop(forStatement &forSmnt);

        bool isOuterMostOklLoop(forStatement &forSmnt,
                                const std::string &attr);

        void setKernelLaunch(functionDeclStatement &kernelSmnt,
                             forStatement &forSmnt,
                             const int kernelIndex);

        void setupHostKernelArgs(functionDeclStatement &kernelSmnt);
        void setupHostHeaders();

        int getInnerLoopLevel(forStatement &forSmnt);

        forStatement* getInnerMostInnerLoop(forStatement &forSmnt);

        exprNode& setDim(token_t *source,
                         const std::string &name,
                         const int index,
                         exprNode *value);

        void splitKernels();

        void splitKernel(functionDeclStatement &kernelSmnt);

        statement_t* extractLoopAsKernel(functionDeclStatement &kernelSmnt,
                                         forStatement &forSmnt,
                                         const int kernelIndex);

        void setupKernels();

        void setupOccaFors(functionDeclStatement &kernelSmnt);

        void addBarriersAfterInnerLoop(forStatement &forSmnt);

        static bool writesToShared(exprNode &expr);

        void replaceOccaFor(forStatement &forSmnt);

        virtual bool usesBarriers();

        virtual std::string getOuterIterator(const int loopIndex) = 0;
        virtual std::string getInnerIterator(const int loopIndex) = 0;
      };
    }
  }
}

#endif
