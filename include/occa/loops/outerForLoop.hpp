#ifndef OCCA_LOOPS_OUTERFORLOOP_HEADER
#define OCCA_LOOPS_OUTERFORLOOP_HEADER

#include <occa/types.hpp>
#include <occa/loops/typelessForLoop.hpp>
#include <occa/loops/innerForLoop.hpp>

namespace occa {
  template <int outerN, class outerIntN>
  class outerForLoop : public typelessForLoop {
  public:
    using innerForLoop1 = innerForLoop<outerN, outerIntN, 1, int>;
    using innerForLoop2 = innerForLoop<outerN, outerIntN, 2, int2>;
    using innerForLoop3 = innerForLoop<outerN, outerIntN, 3, int3>;

    outerForLoop(occa::device device_) :
      typelessForLoop(device_) {}

    outerForLoop(typelessForLoop &other) :
      typelessForLoop(other) {}

    innerForLoop1 inner(occa::iteration innerIteration0) {
      innerIterations = {
        innerIteration0
      };

      return *this;
    }

    innerForLoop2 inner(occa::iteration innerIteration0,
                        occa::iteration innerIteration1) {
      innerIterations = {
        innerIteration0,
        innerIteration1
      };

      return *this;
    }

    innerForLoop3 inner(occa::iteration innerIteration0,
                        occa::iteration innerIteration1,
                        occa::iteration innerIteration2) {
      innerIterations = {
        innerIteration0,
        innerIteration1,
        innerIteration2
      };

      return *this;
    }

    void run(occa::function<void(outerIntN)> fn) {
      typelessRun({}, fn);
    }

    void run(occa::scope scope,
             occa::function<void(outerIntN)> fn) {
      typelessRun(scope, fn);
    }
  };

  typedef outerForLoop<1, int>  outerForLoop1;
  typedef outerForLoop<2, int2> outerForLoop2;
  typedef outerForLoop<3, int3> outerForLoop3;
}

#endif
