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
        // Hack until code-transformation API is done
        static qualifier_t global;

        openclParser(const occa::json &settings_ = occa::json());

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

        void setGlobalQualifiers();
        static void updateGlobalVariables(statement_t *smnt);
        static void addGlobalToFunctionArgs(function_t &func);
        static void addGlobalToVariable(variable_t &var);

        static void updateScopeStructVariables(statement_t *smnt);
        static void addStructToVariable(variable_t &var);
        static void addStructToFunctionArgs(function_t &func);

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
