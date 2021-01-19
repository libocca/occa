#ifndef OCCA_LOOPS_TYPELESSFORLOOP_HEADER
#define OCCA_LOOPS_TYPELESSFORLOOP_HEADER

#include <occa/loops/iteration.hpp>

namespace occa {
  class typelessForLoop {
  public:
    occa::device device;

    occa::iteration outerIterations[3];
    occa::iteration innerIterations[3];
    int outerIterationCount;
    int innerIterationCount;

    typelessForLoop(occa::device device_);

    typelessForLoop(const typelessForLoop &other);

    void typelessRun(const occa::scope &scope,
                     const baseFunction &fn) const;

    occa::scope getForLoopScope(const occa::scope &scope,
                                const baseFunction &fn) const;

    std::string buildOuterLoop(occa::scope &scope,
                               const int index) const;

    std::string buildInnerLoop(occa::scope &scope,
                               const int index) const;

    std::string buildIndexInitializer(const std::string &indexName,
                                      const int iterationCount) const;
  };
}

#endif
