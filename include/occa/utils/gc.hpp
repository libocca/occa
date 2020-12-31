#ifndef OCCA_UTILS_GC_HEADER
#define OCCA_UTILS_GC_HEADER

namespace occa {
  namespace gc {
    // It's more of an internal utility since it's used in occa/internal/utils/gc
    class ringEntry_t {
    public:
      ringEntry_t *leftRingEntry;
      ringEntry_t *rightRingEntry;

      ringEntry_t();

      void removeRef();
      void dontUseRefs();
    };
  }
}

#endif
