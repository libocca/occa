#include <occa/loops/forLoop.hpp>

namespace occa {
  forLoop::forLoop() {}

  forLoop::forLoop(occa::device device_) :
    device(device_) {}

  outerForLoop1 forLoop::outer(occa::iteration outerIteration0) {
    outerForLoop1 loop(device);

    loop.outerIterations = {
      outerIteration0
    };

    return loop;
  }

  outerForLoop2 forLoop::outer(occa::iteration outerIteration0,
                               occa::iteration outerIteration1) {
    outerForLoop2 loop(device);

    loop.outerIterations = {
      outerIteration0,
      outerIteration1
    };

    return loop;
  }

  outerForLoop3 forLoop::outer(occa::iteration outerIteration0,
                               occa::iteration outerIteration1,
                               occa::iteration outerIteration2) {
    outerForLoop3 loop(device);

    loop.outerIterations = {
      outerIteration0,
      outerIteration1,
      outerIteration2
    };

    return loop;
  }

  outerForLoop1 forLoop::tile(occa::tileIteration iteration0) {
    return outer(iteration0);
  }

  outerForLoop2 forLoop::tile(occa::tileIteration iteration0,
                              occa::tileIteration iteration1) {
    return outer(iteration0, iteration1);
  }

  outerForLoop3 forLoop::tile(occa::tileIteration iteration0,
                              occa::tileIteration iteration1,
                              occa::tileIteration iteration2) {
    return outer(iteration0, iteration1, iteration2);
  }
}
