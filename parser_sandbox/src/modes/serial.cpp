#if 0
#include "modes/serial.hpp"

namespace occa {
  namespace lang {
    void serialBackend::transform(statement_t &root,
                                  const properties &props) {
      // Common
      reorderLoops();
      retagOccaLoops();
      splitTileOccaFors();

      // Convert:
      //   - threadIdx
      //   - blockDim
      //   - blockIdx
      //   - gridDim
      setupCudaVariables();

      // Store inner/outer + dim attributes
      // setupOccaVariables();

      // Check conditional barriers
      checkOccaBarriers();

      // Add barriers between for-loops
      //   that use shared memory
      addOccaBarriers();

      // OpenCL needs these
      addFunctionPrototypes();

      // OpenCL uses constant for global
      //   const
      updateConstToConstant();

      // Add for-loops after occaInnerFor#
      addOccaFors();

      // Set kernel launch
      setupOccaFors();

      // Move the defines to the kernel scope
      floatSharedAndExclusivesUp();

      // Setup kernel argument types
      //   const int double -> const int &double
      addArgQualifiers();

      // Setup launch kernel
      loadKernelInfos();

      modifyExclusiveVariables();
    }

    void transformExclusives() {
      context_t &context = root.context;

      statementPtrVector statements = (context
                                       .find("declaration")
                                       .with("attribute", "exclusive"));

      const int sCount = (int) statements.size();
      for (int i = 0; i < sCount; ++i) {
        declarationStatement &s = *(statements[i]);

      }

      statementPtrVector exclusiveStatements =
        (context
         .findStatements(statementType::declaration)
         .hasAttribute("exclusive"));



      const int variableCount = (int) variables.size();
      for (int i = 0; i < variableCount ++i) {
        variable &var = *(variables[i]);
        variable.addArray();
        expressionPtrVector = context.findExpressions(var);
      }
    }
  }
}

#endif
