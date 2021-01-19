#include <occa/loops/forLoop.hpp>

namespace occa {
  forLoop::forLoop() {}

  forLoop::forLoop(occa::device device_) :
    device(device_) {}

  outerForLoop1 forLoop::outer(occa::iteration outerIteration0) {
    outerForLoop1 loop(device);

    loop.outerIterationCount = 1;

    loop.outerIterations[0] = outerIteration0;

    return loop;
  }

  outerForLoop2 forLoop::outer(occa::iteration outerIteration0,
                               occa::iteration outerIteration1) {
    outerForLoop2 loop(device);

    loop.outerIterationCount = 2;

    loop.outerIterations[0] = outerIteration0;
    loop.outerIterations[1] = outerIteration1;

    return loop;
  }

  outerForLoop3 forLoop::outer(occa::iteration outerIteration0,
                               occa::iteration outerIteration1,
                               occa::iteration outerIteration2) {
    outerForLoop3 loop(device);

    loop.outerIterationCount = 3;

    loop.outerIterations[0] = outerIteration0;
    loop.outerIterations[1] = outerIteration1;
    loop.outerIterations[2] = outerIteration2;

    return loop;
  }
}
