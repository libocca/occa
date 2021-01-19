#ifndef OCCA_LOOPS_TYPELESSFORLOOP_HEADER
#define OCCA_LOOPS_TYPELESSFORLOOP_HEADER

#include <occa/functional/array.hpp>
#include <occa/functional/range.hpp>

namespace occa {
  class iteration {
  public:
    inline iteration() {}

    inline iteration(const int) {}

    inline iteration(const range &) {}

    inline iteration(const array<int> &) {}

    inline std::string buildForLoop(const std::string &iterationName) const {
      return "";
    }
  };

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

    std::string buildOuterLoop(const int index) const;

    std::string buildInnerLoop(const int index) const;

    std::string buildIndexInitializer(const std::string &indexName,
                                      const int count) const;
  };
}

#endif
