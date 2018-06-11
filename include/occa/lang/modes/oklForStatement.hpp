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
#ifndef OCCA_TESTS_PARSER_MODES_OKLFORSTATEMENT_HEADER
#define OCCA_TESTS_PARSER_MODES_OKLFORSTATEMENT_HEADER

#include <occa/lang/statement.hpp>

namespace occa {
  namespace lang {
    class exprNode;
    class exprOpNode;
    class leftUnaryOpNode;
    class rightUnaryOpNode;
    class binaryOpNode;

    namespace transforms {
      class smntTreeNode;
    }

    namespace okl {
      class oklForStatement {
      public:
        forStatement &forSmnt;
        const std::string source;
        const bool printErrors;

        variable_t *iterator;
        exprNode *initValue;

        binaryOpNode *checkOp;
        exprNode *checkValue;
        bool checkValueOnRight;
        bool checkIsInclusive;

        exprOpNode *updateOp;
        exprNode *updateValue;
        bool positiveUpdate;

        bool valid;

        oklForStatement(forStatement &forSmnt_,
                        const std::string &source_ = "",
                        const bool printErrors_ = true);

        bool isValid();

        static bool isValid(forStatement &forSmnt_,
                            const std::string &source_ = "",
                            const bool printErrors_ = true);

        bool hasValidInit();

        bool hasValidCheck();

        bool hasValidUpdate();

        bool usesIterator(leftUnaryOpNode &opNode);

        bool usesIterator(rightUnaryOpNode &opNode);

        int usesIterator(binaryOpNode &opNode,
                         exprNode *&value);

        exprNode* getIterationCount();

        exprNode* makeDeclarationValue(exprNode &magicIterator);

        bool isInnerLoop();
        bool isOuterLoop();

        int oklLoopIndex();

        static int oklLoopIndex(forStatement &forSmnt);

        void getOKLLoopPath(statementPtrVector &path);

        static void getOKLLoopPath(forStatement &forSmnt,
                                   statementPtrVector &path);

        std::string sourceStr();

        void printWarning(const std::string &message);
        void printError(const std::string &message);
      };
    }
  }
}

#endif
