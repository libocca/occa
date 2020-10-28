#ifndef OCCA_LANG_MODES_DPCPP_HEADER
#define OCCA_LANG_MODES_DPCPP_HEADER

#include <occa/lang/modes/withLauncherLambda.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class dpcppParser : public withLauncherLambda {
      public:
        qualifier_t device;
	        qualifier_t shared;

        dpcppParser(const occa::properties &settings_ = occa::properties());

        virtual void onClear();
	static bool dumfun(exprNode &smnt){ return true; }
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

        void setKernelQualifiers(functionDeclStatement &kernelSmnt);

        static bool sharedVariableMatcher(exprNode &expr);
      };
    }
  }
}

#endif
