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
#ifndef OCCA_PARSER_MODES_OPENCL_HEADER
#define OCCA_PARSER_MODES_OPENCL_HEADER

#include "occa/lang/parser.hpp"
#include "occa/lang/modes/serial.hpp"
#include "occa/lang/builtins/transforms/finders.hpp"

namespace occa {
  namespace lang {
    namespace okl {
      class openclParser : public parser_t {
      public:
        serialParser hostParser;

        qualifier_t constant;
        qualifier_t kernel;
        qualifier_t global;
        qualifier_t local;

        openclParser(const occa::properties &settings_ = occa::properties());

        //---[ Public ]-----------------
        void writeHostSourceToFile(const std::string &filename) const;
        //==============================

        virtual void onClear();
        virtual void afterParsing();

        void setupHostParser();
        void setupHostHeaders();

        void removeHostOuterLoops(functionDeclStatement &kernelSmnt);

        bool isOuterMostOuterLoop(forStatement &forSmnt);

        void setKernelLaunch(functionDeclStatement &kernelSmnt,
                             forStatement &forSmnt);

        int getInnerLoopLevel(forStatement &forSmnt);

        forStatement* getInnerMostInnerLoop(forStatement &forSmnt);

        exprNode& setDim(token_t *source,
                         const std::string &name,
                         const int index,
                         exprNode *value);

        void addExtensions();

        void updateConstToConstant();

        void setLocalQualifiers();

        void setupKernels();
        void setKernelQualifiers(functionDeclStatement &kernelSmnt);
        void replaceOccaFors(functionDeclStatement &kernelSmnt);
        void replaceOccaFor(forStatement &forSmnt);

        static bool sharedVariableMatcher(exprNode &expr);

        void addFunctionPrototypes();
      };
    }
  }
}

#endif
