#ifndef OCCA_TOOLS_GC_HEADER
#define OCCA_TOOLS_GC_HEADER

#include <stdint.h>
#include <cstddef>
#include <map>
#include <vector>

namespace occa {
  namespace gc {
    class withRefs {
    private:
      int refs;

    public:
      withRefs();

      int getRefs() const;
      void addRef();
      int removeRef();

      void setRefs(const int refs_);
      void dontUseRefs();
    };

    class ringEntry_t {
    public:
      ringEntry_t *leftRingEntry;
      ringEntry_t *rightRingEntry;

      ringEntry_t();

      void removeRef();
      void dontUseRefs();

      bool isAlone() const;
    };

    template <class entry_t>
    class ring_t {
    public:
      bool useRefs;
      ringEntry_t *head;

      ring_t();

      void dontUseRefs();
      void clear();

      void addRef(entry_t *entry);
      void removeRef(entry_t *entry);

      bool needsFree() const;

      int length() const;
    };

    template <class entry_t>
    class multiRing_t {
    public:
      typedef ring_t<entry_t> entryRing_t;
      typedef std::map<entry_t*, entryRing_t> entryRingMap_t;

      bool useRefs;
      entryRingMap_t rings;

      multiRing_t();

      void dontUseRefs();
      void clear();

      void addNewRef(entry_t *entry);
      void removeRef(entry_t *entry);

      bool needsFree() const;
    };
  }
}

#include "gc.tpp"

#endif
