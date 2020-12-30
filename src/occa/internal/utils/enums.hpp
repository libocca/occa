#ifndef OCCA_INTERNAL_UTILS_ENUMS_HEADER
#define OCCA_INTERNAL_UTILS_ENUMS_HEADER

namespace occa {
  namespace sys {
    namespace language {
      static const int notFound = 0;
      static const int CPP      = 1;
      static const int C        = 2;
    }

    enum class CacheLevel {
      L1D,
      L1I,
      L2,
      L3
    };
  }
}

#endif
