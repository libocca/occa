#ifndef OCCA_INTERNAL_LANG_MODES_METAL_HEADER
#define OCCA_INTERNAL_LANG_MODES_METAL_HEADER

#include <occa/internal/lang/modes/withLauncher.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class metalParser : public withLauncher {
       public:
        qualifier_t kernel_q;
        qualifier_t device_q;
        qualifier_t threadgroup_q;
        qualifier_t constant_q;

        metalParser(const occa::json &settings_ = occa::json());

        virtual void onClear() override;
        virtual void beforePreprocessing() override;

        virtual void beforeKernelSplit() override;

        virtual void afterKernelSplit() override;

        virtual std::string getOuterIterator(const int loopIndex) override;

        virtual std::string getInnerIterator(const int loopIndex) override;

        virtual std::string launchBoundsAttribute(const int innerDims[3]) override;

        void setSharedQualifiers();

        void addBarriers();

        void setupHeaders();
        void setupKernels();

        void migrateLocalDecls(functionDeclStatement &kernelSmnt);

        void setKernelQualifiers(function_t &function);
      };
    }
  }
}

#endif
