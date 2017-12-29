#ifndef OCCA_PARSER_MODES_BACKEND_HEADER2
#define OCCA_PARSER_MODES_BACKEND_HEADER2

#include "occa/tools/properties.hpp"
#include "statement.hpp"

namespace occa {
  namespace lang {
    class backend {
    public:
      const properties &props;

      backend(const properties &props_ = "");

      virtual void transform(statement &root) = 0;
    };

    class oklBackend : public backend {
    public:
      oklBackend(const properties &props_);

      virtual void transform(statement &root);
      virtual void backendTransform(statement &root) = 0;

      // @tile(...) -> for-loops
      void splitTileOccaFors(statement &root);

      // @outer -> @outer(#)
      void retagOccaLoops(statement &root);

      void verifyOccaLoop(forStatement &loop);

      // Store inner/outer + dim attributes
      void storeOccaInfo(statement &root);

      // Check conditional barriers
      void checkOccaBarriers(statement &root);

      // Add barriers between for-loops that use shared memory
      void addOccaBarriers(statement &root);

      // Move the defines to the kernel scope
      void floatSharedAndExclusiveDefines(statement &root);
    };
  }
}

#endif
