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
      OCCA_LOOP_INIT_OUTER_INDEX(outerIndex)

      OCCA_LOOP_START_INNER_LOOPS
      OCCA_LOOP_INIT_INNER_INDEX(innerIndex)

      OCCA_LOOP_FUNCTION(outerIndex, innerIndex);

      OCCA_LOOP_END_INNER_LOOPS
      OCCA_LOOP_END_OUTER_LOOPS
    ));
  }

  occa::scope typelessForLoop::getForLoopScope(const occa::scope &scope,
                                               const baseFunction &fn) const {
    occa::scope loopScope = scope;
    if (!loopScope.device.isInitialized()) {
      loopScope.device = device;
    }

    // Inject the function
    loopScope.props["functions/occa_loop_function"] = fn;

    // Setup @outer loops
    std::string outerForLoopsStart, outerForLoopsEnd;
    for (int i = 0; i < outerIterationCount; ++i) {
      outerForLoopsStart += buildOuterLoop(i);
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
        innerForLoopsStart += buildInnerLoop(i);
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

  std::string typelessForLoop::buildOuterLoop(const int index) const {
    return outerIterations[index].buildForLoop(
      "OUTER_INDEX_" + std::to_string(index)
    );
  }

  std::string typelessForLoop::buildInnerLoop(const int index) const {
    return innerIterations[index].buildForLoop(
      "INNER_INDEX_" + std::to_string(index)
    );
  }

  std::string typelessForLoop::buildIndexInitializer(const std::string &indexName,
                                                     const int count) const {
    std::stringstream ss;

    if (innerIterationCount == 1) {
      ss << "const int " << indexName << " = " << indexName << "_1;";
    } else if (innerIterationCount == 2) {
      ss << "int2 " << indexName << ";\n"
         << "" << indexName << ".x = " << indexName << "_1;"
         << "" << indexName << ".y = " << indexName << "_2;";
    } else if (innerIterationCount == 3) {
      ss << "int3 " << indexName << ";\n"
         << "" << indexName << ".x = " << indexName << "_1;"
         << "" << indexName << ".y = " << indexName << "_2;"
         << "" << indexName << ".z = " << indexName << "_3;";
    }

    return ss.str();
  }
}
