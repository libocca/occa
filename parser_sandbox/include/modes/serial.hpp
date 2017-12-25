#ifndef OCCA_PARSER_MODES_SERIAL_HEADER2
#define OCCA_PARSER_MODES_SERIAL_HEADER2

#include "modes/backend.hpp"

namespace occa {
  namespace lang {
    class serialBackend : public oklBackend {
    public:
      virtual void backendTransform(statement &root);

      void setupKernelArgs(statement &root);

      void modifyExclusiveVariables(statement &root);
    };
  }
}

#endif
