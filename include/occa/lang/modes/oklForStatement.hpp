#ifndef OCCA_LANG_MODES_OKLFORSTATEMENT_HEADER
#define OCCA_LANG_MODES_OKLFORSTATEMENT_HEADER

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
