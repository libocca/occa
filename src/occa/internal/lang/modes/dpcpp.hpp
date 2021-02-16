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

        virtual void onClear();

        virtual void beforePreprocessing();
        virtual void beforeKernelSplit();
        virtual void afterKernelSplit();

        virtual std::string getOuterIterator(const int loopIndex);
        virtual std::string getInnerIterator(const int loopIndex);

        // void updateConstToConstant();

        void addExtensions();
        void addBarriers();
        void setupHeaders();
        void setupKernels();
        // void setupAtomics();

        void setFunctionQualifiers();
        void setSharedQualifiers();
        void setKernelQualifiers(function_t &function);
        void migrateLocalDecls(functionDeclStatement &fromSmnt,
                               blockStatement &toSmnt);

        // static bool transformBlockStatement(blockStatement &blockSmnt);
        // static bool transformBasicExpressionStatement(expressionStatement &exprSmnt);

        inline int dpcppDimensionOrder(const int index) { return 2 - index; }
      };
    } // namespace okl
  }   // namespace lang
} // namespace occa

#endif
