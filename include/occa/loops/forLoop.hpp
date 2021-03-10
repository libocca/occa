#ifndef OCCA_LOOPS_FORLOOP_HEADER
#define OCCA_LOOPS_FORLOOP_HEADER

#include <occa/loops/outerForLoop.hpp>
#include <occa/loops/innerForLoop.hpp>

namespace occa {
  class forLoop {
  public:
    occa::device device;

    forLoop();

    forLoop(occa::device device_);

    outerForLoop1 outer(occa::iteration iteration0);

    outerForLoop2 outer(occa::iteration iteration0,
                        occa::iteration iteration1);

    outerForLoop3 outer(occa::iteration iteration0,
                        occa::iteration iteration1,
                        occa::iteration iteration2);

    outerForLoop1 tile(occa::tileIteration iteration0);

    outerForLoop2 tile(occa::tileIteration iteration0,
                       occa::tileIteration iteration1);

    outerForLoop3 tile(occa::tileIteration iteration0,
                       occa::tileIteration iteration1,
                       occa::tileIteration iteration2);
  };
}

#endif
