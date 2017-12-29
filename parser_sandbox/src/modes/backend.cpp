#include "modes/backend.hpp"

namespace occa {
  namespace lang {
    backend::backend(const properties &props_) :
      props(props_) {}

    void oklBackend::transform(statement &root) {
      // @tile(...) -> for-loops
      splitTileOccaFors(root);

      // @outer -> @outer(#)
      retagOccaLoops(root);

      // Store inner/outer + dim attributes
      storeOccaInfo(root);

      // Check conditional barriers
      checkOccaBarriers(root);

      // Add barriers between for-loops
      //   that use shared memory
      addOccaBarriers(root);

      // Move the defines to the root scope
      floatSharedAndExclusiveDefines(root);

      backendTransform(root);
    }

    // @tile(root) -> for-loops
    void oklBackend::splitTileOccaFors(statement &root) {
#if 0
      statementQuery tiledQuery = (query::findForLoops()
                                   .withAttribute("tile"));

      statementPtrVector tiledLoops = tiledQuery(root);
      const int loopCount = (int) tiledLoops.size();
      for (int i = 0; i < loopCount; ++i) {
        forStatement &loop = tiledLoops[i]->to<forStatement>();
        tileAttribute &tile = loop.getAttribute<tileAttribute>("tile");

        verifyOccaLoop(loop);
      }
#endif
    }

    // @outer -> @outer(#)
    void oklBackend::retagOccaLoops(statement &root) {
#if 0
      statementQuery outerLoops = (query::findForLoops(query::first)
                                   .withAttribute("outer"));
      statementPtrVector loops = outerLoops(kernel);
      statementPtrVector nextLoops;

      while (loops.size()) {
        const int loopCount = (int) loops.size();
        for (int i = 0; i < loopCount; ++i) {
          nextLoops = outerLoops(*(loops[i]))
        }
      }
#endif
    }

    void oklBackend::verifyOccaLoop(forStatement &loop) {
#if 0
      const std::string initError = ("@outer, @inner, and @tile loops must have a simple"
                                     " variable declaration statement (e.g. int x = 0)");
      if (loop.init.type() != statementType::expression) {
        loop.init.error(initError);
      }
      expressionStatement &init = loop.init.to<expressionStatement>();
      type_t *initType;
      variable *initVar;
      expression *initExpression;
      if (!query(init.expression)
          .hasFormat(query::type(initType, query::flags::optional)
                     + query::variable(initVar)
                     + query::op("=")
                     + query::expression(initExpression))) {
        loop.init.error(initError);
      }
#endif
    }

    // Store inner/outer + dim attributes
    void oklBackend::storeOccaInfo(statement &root) {
    }

    // Check conditional barriers
    void oklBackend::checkOccaBarriers(statement &root) {
    }

    // Add barriers between for-loops that use shared memory
    void oklBackend::addOccaBarriers(statement &root) {
    }

      // Move the defines to the kernel scope
    void oklBackend::floatSharedAndExclusiveDefines(statement &root) {
    }
  }
}
