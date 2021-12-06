#ifndef OCCA_LANG_MODES_DPCPP_HEADER
#define OCCA_LANG_MODES_DPCPP_HEADER
#include <occa/internal/lang/modes/withLauncher.hpp>

namespace occa
{
  namespace lang
  {
    const primitive_t syclQueue     ("sycl::queue");
    const primitive_t syclNdRange   ("sycl::nd_range<3>");
    const primitive_t syclNdItem    ("sycl::nd_item<3>");
    const primitive_t syclHandler   ("sycl::handler");
    const primitive_t syclAccessor  ("auto");

    namespace okl
    {
      class dpcppParser : public withLauncher
      {
      public:
        qualifier_t kernel;
        qualifier_t device;
        qualifier_t shared;

        dpcppParser(const occa::json &settings_ = occa::json());

        virtual void onClear() override;

        virtual void beforePreprocessing() override;
        virtual void beforeKernelSplit() override;
        virtual void afterKernelSplit() override;

        virtual std::string getOuterIterator(const int loopIndex) override;
        virtual std::string getInnerIterator(const int loopIndex) override;
        virtual std::string launchBoundsAttribute(const int innerDims[3]) override;

        void addExtensions();
        void addBarriers();
        void setupHeaders();
        void setupKernels();

        void setFunctionQualifiers();
        void setSharedQualifiers();
        void setKernelQualifiers(function_t &function);
        void migrateLocalDecls(functionDeclStatement &kernelSmnt);
        void setLaunchBounds();

        void setupAtomics();
        static bool transformAtomicBlockStatement(blockStatement &blockSmnt);
        static bool transformAtomicBasicExpressionStatement(expressionStatement &exprSmnt);

      private:
        inline int dpcppDimensionOrder(const int index) { return 2 - index; }
      };
    } // namespace okl
  }   // namespace lang
} // namespace occa

#endif
