#ifndef OCCA_INTERNAL_UTILS_UVA_HEADER
#define OCCA_INTERNAL_UTILS_UVA_HEADER

#include <occa/internal/io/output.hpp>

namespace occa {
  namespace uvaFlag {
    static const int none      = 0;
    static const int isManaged = (1 << 0);
    static const int inDevice  = (1 << 1);
    static const int isStale   = (1 << 2);
  }

  class device;
  class memory;
  class modeMemory_t;
  class ptrRange;

  typedef std::map<ptrRange, occa::modeMemory_t*> ptrRangeMap;
  typedef std::vector<occa::modeMemory_t*>        memoryVector;

  extern ptrRangeMap uvaMap;
  extern memoryVector uvaStaleMemory;

  //---[ ptrRange ]---------------------
  class ptrRange {
  public:
    char *start, *end;

    ptrRange();
    ptrRange(void *ptr, const udim_t bytes = 0);
    ptrRange(const ptrRange &other);

    ptrRange& operator =  (const ptrRange &other);
    bool operator == (const ptrRange &other) const;
    bool operator != (const ptrRange &other) const;
  };

  int operator < (const ptrRange &a,
                  const ptrRange &b);

  std::ostream& operator << (std::ostream& out,
                           const ptrRange &range);
  //====================================

  //---[ UVA ]--------------------------
  void removeFromStaleMap(void *ptr);
  void removeFromStaleMap(modeMemory_t *mem);
  //====================================
}

#endif
