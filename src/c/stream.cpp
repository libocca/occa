#include <occa/internal/c/types.hpp>
#include <occa/c/device.h>
#include <occa/c/dtype.h>

OCCA_START_EXTERN_C

void occaStreamFinish(occaStream stream) {
  occa::c::stream(stream).finish();
}

OCCA_END_EXTERN_C
