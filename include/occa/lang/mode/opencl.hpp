#ifndef OCCA_LANG_MODES_OPENCL_HEADER
#define OCCA_LANG_MODES_OPENCL_HEADER

#include <occa/lang/mode/withLauncher.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class openclParser : public withLauncher {
      public:
        qualifier_t constant;
        qualifier_t kernel;
        qualifier_t global;
        qualifier_t local;

        openclParser(const occa::properties &settings_ = occa::properties());

        virtual void onClear();

        virtual void beforePreprocessing();

        virtual void beforeKernelSplit();

        virtual void afterKernelSplit();

        virtual std::string getOuterIterator(const int loopIndex);

        virtual std::string getInnerIterator(const int loopIndex);

        void addExtensions();

        void updateConstToConstant();

        void setLocalQualifiers();

        static bool sharedVariableMatcher(exprNode &expr);

        void addBarriers();

        void addFunctionPrototypes();

        void setupKernels();

        void migrateLocalDecls(functionDeclStatement &kernelSmnt);

        void setKernelQualifiers(function_t &function);
      };
    }
  }
}

#endif
