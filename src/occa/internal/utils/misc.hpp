#ifndef OCCA_INTERNAL_UTILS_MISC_HEADER
#define OCCA_INTERNAL_UTILS_MISC_HEADER

#include <initializer_list>

#include <occa/defines.hpp>
#include <occa/types.hpp>

namespace occa {
  udim_t ptrDiff(void *start, void *end);

  template <class TM>
  TM min(std::initializer_list<TM> values) {
    TM minValue = *values.begin();
    for (const TM &value : values) {
      if (value < minValue) {
        minValue = value;
      }
    }
    return minValue;
  }

  template <class TM>
  TM max(std::initializer_list<TM> values) {
    TM maxValue = *values.begin();
    for (const TM &value : values) {
      if (maxValue < value) {
        maxValue = value;
      }
    }
    return maxValue;
  }

  template <class TM>
  void ignoreResult(const TM &t) {
    (void) t;
  }
}

#endif
