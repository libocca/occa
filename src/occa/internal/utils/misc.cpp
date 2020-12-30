#include <occa/internal/utils/misc.hpp>

namespace occa {
  udim_t ptrDiff(void *start, void *end) {
    if (start < end) {
      return (udim_t) (((char*) end) - ((char*) start));
    }
    return (udim_t) (((char*) start) - ((char*) end));
  }
}
