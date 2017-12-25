#ifndef OCCA_PARSER_MODES_OPENCL_HEADER2
#define OCCA_PARSER_MODES_OPENCL_HEADER2

#include "modes/backend.hpp"

namespace occa {
  namespace lang {
    class openclBackend : public oklBackend {
    public:
      virtual void backendTransform(statement &root);

      void addFunctionPrototypes(statement &root);

      void updateConstToConstant(statement &root);

      void addOccaFors(statement &root);

      void setupKernelArgs(statement &root);

      void setupLaunchKernel(statement &root);
    };
  }
}

#endif
