#ifndef OCCA_LANG_MODES_SERIAL_HEADER
#define OCCA_LANG_MODES_SERIAL_HEADER

#include <occa/lang/parser.hpp>

namespace occa {
  namespace lang {
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
        void setupExclusiveDeclaration(declarationStatement &declSmnt);
        void setupExclusiveIndices();

        void defineExclusiveVariableAsArray(variable_t &var);

        exprNode* addExclusiveVariableArrayAccessor(statement_t &smnt,
                                                    exprNode &expr,
                                                    variable_t &var);
      };
    }
  }
}

#endif
