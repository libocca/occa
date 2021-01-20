#ifndef OCCA_LOOPS_FORLOOP_HEADER
#define OCCA_LOOPS_FORLOOP_HEADER

#include <occa/loops/outerForLoop.hpp>

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
  };
}

#endif
