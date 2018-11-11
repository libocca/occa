#ifndef OCCA_TOOLS_MISC_HEADER
#define OCCA_TOOLS_MISC_HEADER

#include <occa/defines.hpp>
#include <occa/types.hpp>

namespace occa {
  udim_t ptrDiff(void *start, void *end);

  template <class TM>
  void ignoreResult(const TM &t) {
    (void) t;
  }
}

#endif
