#include <map>

#include <occa/core/base.hpp>
#include <occa/internal/utils/misc.hpp>
#include <occa/internal/utils/uva.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  ptrRangeMap uvaMap;
  memoryVector uvaStaleMemory;

  //---[ ptrRange ]---------------------
  ptrRange::ptrRange() :
    start(NULL),
    end(NULL) {}

  ptrRange::ptrRange(void *ptr, const udim_t bytes) :
    start((char*) ptr),
    end(((char*) ptr) + bytes) {}

  ptrRange::ptrRange(const ptrRange &other) :
    start(other.start),
    end(other.end) {}

  ptrRange& ptrRange::operator = (const ptrRange &other) {
    start = other.start;
    end   = other.end;

    return *this;
  }

  bool ptrRange::operator == (const ptrRange &other) const {
    return ((start < other.end) &&
            (end > other.start));
  }

  bool ptrRange::operator != (const ptrRange &other) const {
    return ((start >= other.end) ||
            (end <= other.start));
  }

  int operator < (const ptrRange &a, const ptrRange &b) {
    return ((a != b) && (a.start < b.start));
  }

  std::ostream& operator << (std::ostream& out,
                             const ptrRange &range) {
    out << '['
        << (void*) range.start << ", " << (void*) range.end
        << ']';
    return out;
  }
  //====================================


  //---[ UVA ]--------------------------
  void removeFromStaleMap(void *ptr) {
    ptrRangeMap::iterator it = uvaMap.find(ptr);
    if (it == uvaMap.end()) {
      return;
    }

    memory m(it->second);
    if (!m.uvaIsStale()) {
      return;
    }

    removeFromStaleMap(m.getModeMemory());
  }

  void removeFromStaleMap(modeMemory_t *mem) {
    if (!mem) {
      return;
    }

    occa::memory m(mem);
    const size_t staleEntries = uvaStaleMemory.size();

    for (size_t i = 0; i < staleEntries; ++i) {
      if (uvaStaleMemory[i] == mem) {
        m.uvaMarkFresh();
        uvaStaleMemory.erase(uvaStaleMemory.begin() + i);
        break;
      }
    }
  }
  //====================================
}
