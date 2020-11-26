#ifndef OCCA_LANG_MODES_CUDA_HEADER
#define OCCA_LANG_MODES_CUDA_HEADER

#include <occa/lang/modes/withLauncher.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class cudaParser : public withLauncher {
      public:
        qualifier_t constant;
        qualifier_t global;
        qualifier_t device;
        qualifier_t shared;

        cudaParser(const occa::properties &settings_ = occa::properties());

        virtual void onClear();

        virtual void beforePreprocessing();

        virtual void beforeKernelSplit();

        virtual void afterKernelSplit();

        virtual std::string getOuterIterator(const int loopIndex);

        virtual std::string getInnerIterator(const int loopIndex);

        void updateConstToConstant();

        void setFunctionQualifiers();

        void setSharedQualifiers();

        void addBarriers();

        void setupKernels();

        void setupAtomics();

        static bool transformBlockStatement(blockStatement &blockSmnt);

        static bool transformBasicExpressionStatement(expressionStatement &exprSmnt);
      };
    }
  }
}

#endif
