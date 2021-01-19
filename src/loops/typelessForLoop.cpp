#include <occa/loops/typelessForLoop.hpp>
#include <occa/experimental/kernelBuilder.hpp>

namespace occa {
  typelessForLoop::typelessForLoop(occa::device device_) :
    device(device_),
    outerIterationCount(0),
    innerIterationCount(0) {}

  typelessForLoop::typelessForLoop(const typelessForLoop &other) :
    device(other.device),
    outerIterations(other.outerIterations),
    innerIterations(other.innerIterations),
    outerIterationCount(other.outerIterationCount),
    innerIterationCount(other.innerIterationCount) {}

  void typelessForLoop::typelessRun(const occa::scope &scope,
                                    const baseFunction &fn) const {
    OCCA_JIT(getForLoopScope(scope, fn), (
      OCCA_LOOP_START_OUTER_LOOPS
      OCCA_LOOP_START_INNER_LOOPS

      OCCA_LOOP_INIT_OUTER_INDEX(outerIndex)
      OCCA_LOOP_INIT_INNER_INDEX(innerIndex)

      OCCA_LOOP_FUNCTION(outerIndex, innerIndex);

      OCCA_LOOP_END_INNER_LOOPS
      OCCA_LOOP_END_OUTER_LOOPS
    ));
  }

  occa::scope typelessForLoop::getForLoopScope(const occa::scope &scope,
                                               const baseFunction &fn) const {
    occa::scope loopScope = scope + fn.scope;
    if (!loopScope.device.isInitialized()) {
      loopScope.device = device;
    }

    // Temporary until issue #175 is resolved
    loopScope.props["okl/validate"] = false;

    // Inject the function
    loopScope.props["functions/occa_loop_function"] = fn;

    // Setup @outer loops
    std::string outerForLoopsStart, outerForLoopsEnd;
    for (int i = 0; i < outerIterationCount; ++i) {
      outerForLoopsStart += buildOuterLoop(loopScope, i);
      outerForLoopsEnd += "}";
    }
    loopScope.props["defines/OCCA_LOOP_START_OUTER_LOOPS"] = outerForLoopsStart;
    loopScope.props["defines/OCCA_LOOP_END_OUTER_LOOPS"] = outerForLoopsEnd;

    loopScope.props["defines/OCCA_LOOP_INIT_OUTER_INDEX(OUTER_INDEX)"] = (
      buildIndexInitializer("OUTER_INDEX", outerIterationCount)
    );

    if (innerIterationCount) {
      // Setup @inner loops
      std::string innerForLoopsStart, innerForLoopsEnd;
      for (int i = 0; i < innerIterationCount; ++i) {
        innerForLoopsStart += buildInnerLoop(loopScope, i);
        innerForLoopsEnd += "}";
      }
      loopScope.props["defines/OCCA_LOOP_START_INNER_LOOPS"] = innerForLoopsStart;
      loopScope.props["defines/OCCA_LOOP_END_INNER_LOOPS"] = innerForLoopsEnd;

      loopScope.props["defines/OCCA_LOOP_INIT_INNER_INDEX(INNER_INDEX)"] = (
        buildIndexInitializer("INNER_INDEX", innerIterationCount)
      );
    } else {
      // Nothing to setup for @inner loops
      loopScope.props["defines/OCCA_LOOP_START_INNER_LOOPS"] = "";
      loopScope.props["defines/OCCA_LOOP_END_INNER_LOOPS"] = "";
      loopScope.props["defines/OCCA_LOOP_INIT_INNER_INDEX(INDEX)"] = "";
    }

    // Define function call
    strVector argumentValues;
    if (innerIterationCount) {
      argumentValues = {"OUTER_INDEX", "INNER_INDEX"};
    } else {
      argumentValues = {"OUTER_INDEX"};
    }
    loopScope.props["defines/OCCA_LOOP_FUNCTION(OUTER_INDEX, INNER_INDEX)"] = (
      fn.buildFunctionCall("occa_loop_function", argumentValues)
    );

    return loopScope;
  }

  std::string typelessForLoop::buildOuterLoop(occa::scope &scope,
                                              const int index) const {
    return "@outer " + outerIterations[index].buildForLoop(
      scope,
      "OUTER_INDEX_" + std::to_string(index)
    );
  }

  std::string typelessForLoop::buildInnerLoop(occa::scope &scope,
                                              const int index) const {
    return "@inner " + innerIterations[index].buildForLoop(
      scope,
      "INNER_INDEX_" + std::to_string(index)
    );
  }

  std::string typelessForLoop::buildIndexInitializer(const std::string &indexName,
                                                     const int iterationCount) const {
    std::stringstream ss;

    if (iterationCount == 1) {
      ss << "const int " << indexName << " = " << indexName << "_0;";
    } else if (iterationCount == 2) {
      ss << "int2 " << indexName << ";"
         << " " << indexName << ".x = " << indexName << "_0;"
         << " " << indexName << ".y = " << indexName << "_1;";
    } else if (iterationCount == 3) {
      ss << "int3 " << indexName << "; "
         << " " << indexName << ".x = " << indexName << "_0;"
         << " " << indexName << ".y = " << indexName << "_1;"
         << " " << indexName << ".z = " << indexName << "_2;";
    }

    return ss.str();
  }
}
