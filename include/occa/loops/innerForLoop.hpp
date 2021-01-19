#ifndef OCCA_LOOPS_INNERFORLOOP_HEADER
#define OCCA_LOOPS_INNERFORLOOP_HEADER

#include <occa/loops/typelessForLoop.hpp>

namespace occa {
  template <int outerN, class outerIntN,
            int innerN, class innerIntN>
  class innerForLoop : public typelessForLoop {
  public:
    innerForLoop(typelessForLoop &other) :
      typelessForLoop(other) {}

    void run(occa::function<void(outerIntN, innerIntN)> fn) {
      typelessRun({}, fn);
    }

    void run(occa::scope scope,
             occa::function<void(outerIntN, innerIntN)> fn) {
      typelessRun(scope, fn);
    }
  };
}

#endif
