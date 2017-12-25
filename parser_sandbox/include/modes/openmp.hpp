#ifndef OCCA_PARSER_MODES_OPENMP_HEADER2
#define OCCA_PARSER_MODES_OPENMP_HEADER2

#include "modes/serial.hpp"

namespace occa {
  namespace lang {
    class openmpBackend : public serialBackend {
    public:
      virtual void backendTransform(statement &root);

      void addPragmas(statement &root);
    };
  }
}

#endif
