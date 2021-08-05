#ifndef OCCA_LANG_MODES_DPCPP_HEADER
#define OCCA_LANG_MODES_DPCPP_HEADER
#include <occa/internal/lang/modes/withLauncher.hpp>

namespace occa
{
  namespace lang
  {
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

        void addExtensions();
        void addBarriers();
        void setupHeaders();
        void setupKernels();

        void setFunctionQualifiers();
        void setSharedQualifiers();
        void setKernelQualifiers(function_t &function);
        void migrateLocalDecls(functionDeclStatement &kernelSmnt);

      void setupAtomics();
      static bool transformAtomicBlockStatement(blockStatement &blockSmnt);
      static bool transformAtomicBasicExpressionStatement(expressionStatement &exprSmnt);

      private:
        inline static const std::string sycl_header{"CL/sycl.hpp"};
        inline static const std::string queue_name{"queue_"};
        inline static const std::string ndrange_name{"range_"};
        inline static const std::string group_handler_name{"handler_"};
        inline static const std::string work_item_name{"item_"};

        inline int dpcppDimensionOrder(const int index) { return 2 - index; }
      };
    } // namespace okl
  }   // namespace lang
} // namespace occa

#endif
