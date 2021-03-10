#include <occa/loops/typelessForLoop.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/experimental/kernelBuilder.hpp>

namespace occa {
  typelessForLoop::typelessForLoop(occa::device device_) :
    device(device_) {}

  typelessForLoop::typelessForLoop(const typelessForLoop &other) :
    device(other.device),
    outerIterations(other.outerIterations),
    innerIterations(other.innerIterations) {}

  void typelessForLoop::typelessRun(const occa::scope &scope,
                                    const baseFunction &fn) const {
    OCCA_JIT(getForLoopScope(scope, fn), (
      OCCA_LOOP_START_OUTER_LOOPS
      OCCA_LOOP_START_INNER_LOOPS

      OCCA_LOOP_INIT_OUTER_INDEX
      OCCA_LOOP_INIT_INNER_INDEX

      OCCA_LOOP_FUNCTION

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

    const int outerIterationCount = (int) outerIterations.size();
    const int innerIterationCount = (int) innerIterations.size();

    // Inject the function information
    const functionDefinition &fnDefinition = fn.definition();

    loopScope.props["defines/OCCA_LOOP_FUNCTION"] = fnDefinition.bodySource;

    // TODO: This is a hack, we should really be parsing the content
    //       and finding the argument names
    strVector arguments = split(fnDefinition.argumentSource, ',');

    const std::string outerIndexName = strip(
      split(arguments[0], ' ').back()
    );
    loopScope.props["defines/OCCA_LOOP_OUTER_INDEX_NAME"] = (
      outerIndexName.size()
      ? outerIndexName
      : "_loopOuterIndex"
    );

    if (innerIterationCount) {
      const std::string innerIndexName = strip(
        split(arguments[1], ' ').back()
      );
      loopScope.props["defines/OCCA_LOOP_INNER_INDEX_NAME"] = (
        innerIndexName.size()
        ? innerIndexName
        : "_loopInnerIndex"
      );
    }

    // Setup @outer loops
    std::string outerForLoopsStart, outerForLoopsEnd;
    for (int i = 0; i < outerIterationCount; ++i) {
      outerForLoopsStart += buildOuterLoop(loopScope, i);
      outerForLoopsEnd += "}";
    }
    loopScope.props["defines/OCCA_LOOP_START_OUTER_LOOPS"] = outerForLoopsStart;
    loopScope.props["defines/OCCA_LOOP_END_OUTER_LOOPS"] = outerForLoopsEnd;

    loopScope.props["defines/OCCA_LOOP_INIT_OUTER_INDEX"] = (
      buildIndexInitializer("OCCA_LOOP_OUTER_INDEX_NAME",
                            "OUTER_INDEX",
                            outerIterationCount)
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

      loopScope.props["defines/OCCA_LOOP_INIT_INNER_INDEX"] = (
        buildIndexInitializer("OCCA_LOOP_INNER_INDEX_NAME",
                              "INNER_INDEX",
                              innerIterationCount)
      );
    } else {
      // Nothing to setup for @inner loops
      loopScope.props["defines/OCCA_LOOP_START_INNER_LOOPS"] = "";
      loopScope.props["defines/OCCA_LOOP_END_INNER_LOOPS"] = "";
      loopScope.props["defines/OCCA_LOOP_INIT_INNER_INDEX"] = "";
    }

    return loopScope;
  }

  std::string typelessForLoop::buildOuterLoop(occa::scope &scope,
                                              const int index) const {
    return outerIterations[index].buildForLoop(
      forLoopType::outer,
      scope,
      "OUTER_INDEX_" + std::to_string(index)
    );
  }

  std::string typelessForLoop::buildInnerLoop(occa::scope &scope,
                                              const int index) const {
    return innerIterations[index].buildForLoop(
      forLoopType::inner,
      scope,
      "INNER_INDEX_" + std::to_string(index)
    );
  }

  std::string typelessForLoop::buildIndexInitializer(const std::string &indexName,
                                                     const std::string &loopIndexNamePrefix,
                                                     const int iterationCount) const {
    std::stringstream ss;

    if (iterationCount == 1) {
      ss << "const int " << indexName << " = " << loopIndexNamePrefix << "_0;";
    } else if (iterationCount == 2) {
      ss << "int2 " << indexName << ";"
         << " " << indexName << ".x = " << loopIndexNamePrefix << "_0;"
         << " " << indexName << ".y = " << loopIndexNamePrefix << "_1;";
    } else if (iterationCount == 3) {
      ss << "int3 " << indexName << "; "
         << " " << indexName << ".x = " << loopIndexNamePrefix << "_0;"
         << " " << indexName << ".y = " << loopIndexNamePrefix << "_1;"
         << " " << indexName << ".z = " << loopIndexNamePrefix << "_2;";
    }

    return ss.str();
  }
}
