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
#ifndef OCCA_TESTS_PARSER_MODES_OKL_HEADER
#define OCCA_TESTS_PARSER_MODES_OKL_HEADER

#include <vector>

#include <occa/lang/statement.hpp>

namespace occa {
  namespace lang {
    class leftUnaryOpNode;
    class rightUnaryOpNode;
    class binaryOpNode;

    namespace transforms {
      class smntTreeNode;
    }

    namespace okl {
      bool checkKernels(statement_t &root);

      bool checkKernel(functionDeclStatement &kernelSmnt);

      //---[ Declaration ]--------------
      bool checkLoops(functionDeclStatement &kernelSmnt);

      bool checkForDoubleLoops(statementPtrVector &loopSmnts,
                               const std::string &badAttr);

      bool checkOklForStatements(functionDeclStatement &kernelSmnt,
                                 statementPtrVector &forSmnts,
                                 const std::string &attrName);
      //================================

      //---[ Loop Logic ]---------------
      bool oklLoopMatcher(statement_t &smnt);
      bool oklDeclAttrMatcher(statement_t &smnt,
                              const std::string &attr);
      bool oklAttrMatcher(statement_t &smnt,
                          const std::string &attr);
      bool oklSharedMatcher(statement_t &smnt);
      bool oklExclusiveMatcher(statement_t &smnt);

      bool checkLoopOrders(functionDeclStatement &kernelSmnt);

      bool checkLoopOrder(transforms::smntTreeNode &root);
      bool checkLoopType(transforms::smntTreeNode &node,
                         int &outerCount,
                         int &innerCount);
      //================================

      //---[ Type Logic ]---------------
      bool checkSharedOrder(transforms::smntTreeNode &root);
      bool checkExclusiveOrder(transforms::smntTreeNode &root);
      bool checkOKLTypeInstance(statement_t &typeSmnt,
                                const std::string &attr);
      bool checkValidSharedArray(statement_t &smnt);
      //================================

      //---[ Skip Logic ]---------------
      bool checkBreakAndContinue(functionDeclStatement &kernelSmnt);
      //================================

      //---[ Transformations ]----------
      void addAttributes(parser_t &parser);

      void setLoopIndices(functionDeclStatement &kernelSmnt);

      void setForLoopIndex(forStatement &forSmnt,
                           const std::string &attr);
      //================================
    }
  }
}

#endif
