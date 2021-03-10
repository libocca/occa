#ifndef OCCA_INTERNAL_LANG_MODES_OKLFORSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_MODES_OKLFORSTATEMENT_HEADER

#include <occa/internal/lang/statement.hpp>

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

        std::string oklAttr;

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

        static int getOklLoopIndex(forStatement &forSmnt, const std::string &oklAttr);

        statementArray getOklLoopPath();

        static statementArray getOklLoopPath(forStatement &forSmnt);

        std::string sourceStr();

        void printWarning(const std::string &message);
        void printError(const std::string &message);
      };
    }
  }
}

#endif
