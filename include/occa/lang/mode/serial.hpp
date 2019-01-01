#ifndef OCCA_LANG_MODES_SERIAL_HEADER
#define OCCA_LANG_MODES_SERIAL_HEADER

#include <occa/lang/parser.hpp>
#include <occa/lang/transforms/builtins/finders.hpp>

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
