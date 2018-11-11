#include <iostream>

namespace occa {
  namespace gc {
    template <class entry_t>
    ring_t<entry_t>::ring_t() :
      useRefs(true),
      head(NULL) {}

    template <class entry_t>
    void ring_t<entry_t>::dontUseRefs() {
      useRefs = false;
    }

    template <class entry_t>
    void ring_t<entry_t>::clear() {
      useRefs = true;
      head = NULL;
    }

    template <class entry_t>
    void ring_t<entry_t>::addRef(entry_t *entry) {
      if (!entry) {
        return;
      }
      entry->removeRef();
      if (!head) {
        head = entry;
        return;
      }
      ringEntry_t *tail = head->leftRingEntry;
      entry->leftRingEntry  = tail;
      tail->rightRingEntry  = entry;
      head->leftRingEntry   = entry;
      entry->rightRingEntry = head;
    }

    template <class entry_t>
    void ring_t<entry_t>::removeRef(entry_t *entry) {
      if (!entry || !head) {
        return;
      }
      ringEntry_t *tail = head->leftRingEntry;
      entry->removeRef();
      if (head == entry) {
        head = ((tail != entry)
                ? tail
                : NULL);
      }
    }

    template <class entry_t>
    bool ring_t<entry_t>::needsFree() const {
      return useRefs && (head == NULL);
    }

    //---[ multiRing_t ]----------------

    template <class entry_t>
    multiRing_t<entry_t>::multiRing_t() :
      useRefs(true) {}

    template <class entry_t>
    void multiRing_t<entry_t>::dontUseRefs() {
      useRefs = false;
    }

    template <class entry_t>
    void multiRing_t<entry_t>::clear() {
      useRefs = true;
      rings.clear();
    }

    template <class entry_t>
    void multiRing_t<entry_t>::addNewRef(entry_t *entry) {
      if (!entry) {
        return;
      }
      rings[entry].addRef(entry);
    }

    template <class entry_t>
    void multiRing_t<entry_t>::removeRef(entry_t *entry) {
      if (!entry) {
        return;
      }
      typename entryRingMap_t::iterator it = rings.find(entry);
      if (it == rings.end()) {
        ring_t<entry_t> &ring = it->second;
        ring.removeRef(entry);
        rings.erase(it);
        // Change key if head changed
        if (ring.head &&
            ((entry_t*) ring.head != entry)) {
          rings[ring.head] = ring;
        }

      } else {
        entry->removeRef();
      }
    }

    template <class entry_t>
    bool multiRing_t<entry_t>::needsFree() const {
      return useRefs && rings.size();
    }
  }
}
