#ifndef OCCA_INTERNAL_LANG_MODES_OPENCL_HEADER
#define OCCA_INTERNAL_LANG_MODES_OPENCL_HEADER

#include <occa/internal/lang/modes/withLauncher.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class openclParser : public withLauncher {
       public:
        qualifier_t constant;
        qualifier_t kernel;
        qualifier_t local;
        qualifier_t global;

        openclParser(const occa::json &settings_ = occa::json());

        virtual void onClear() override;
        virtual void beforePreprocessing() override;

        virtual void beforeKernelSplit() override;

        virtual void afterKernelSplit() override;

        virtual std::string getOuterIterator(const int loopIndex) override;

        virtual std::string getInnerIterator(const int loopIndex) override;

        virtual std::string launchBoundsAttribute(const int innerDims[3]) override;

        void addExtensions();

        void updateConstToConstant();

        void setLocalQualifiers();

        void setGlobalQualifiers();

        void addStructToVariable(variable_t &var);
        void addStructToFunctionArgs(function_t &func);

        void addBarriers();

        void addFunctionPrototypes();

        void addStructQualifiers();

        void setupKernels();

        void migrateLocalDecls(functionDeclStatement &kernelSmnt);

        void setKernelQualifiers(function_t &function);
      };
    }
  }
}

#endif
