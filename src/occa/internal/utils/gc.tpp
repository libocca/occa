#include <iostream>

namespace occa {
  namespace gc {
   #if OCCA_THREAD_SHARABLE_ENABLED
    template <class entry_t>
    mutex_t ring_t<entry_t>::mutex;
   #endif
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
     #if OCCA_THREAD_SHARABLE_ENABLED
      mutex.lock();
     #endif
      if (!entry || head == entry) {
       #if OCCA_THREAD_SHARABLE_ENABLED
        mutex.unlock();
       #endif
        return;
      }
      entry->removeRef();
      if (!head) {
        head = entry;
       #if OCCA_THREAD_SHARABLE_ENABLED
        mutex.unlock();
       #endif
        return;
      }
      ringEntry_t *tail = head->leftRingEntry;
      entry->leftRingEntry  = tail;
      tail->rightRingEntry  = entry;
      head->leftRingEntry   = entry;
      entry->rightRingEntry = head;
     #if OCCA_THREAD_SHARABLE_ENABLED
      mutex.unlock();
     #endif
    }

   #if OCCA_THREAD_SHARABLE_ENABLED
    template <class entry_t>
    void ring_t<entry_t>::removeRef(entry_t *entry, const bool threadLock) {
      if (threadLock)
        mutex.lock();
      // Check if the ring is empty
      if (!entry || !head) {
        mutex.unlock();
        return;
      }
      ringEntry_t *tail = head->leftRingEntry;
      // Remove the entry ref from its ring
      entry->removeRef();
      if (head == entry) {
        // Change the head to the tail if entry happened to be the old head
        head = ((tail != entry)
                ? tail
                : NULL);
      }
      mutex.unlock();
    }
   #else
    template <class entry_t>
    void ring_t<entry_t>::removeRef(entry_t *entry) {
      // Check if the ring is empty
      if (!entry || !head) {
        return;
      }
      ringEntry_t *tail = head->leftRingEntry;
      // Remove the entry ref from its ring
      entry->removeRef();
      if (head == entry) {
        // Change the head to the tail if entry happened to be the old head
        head = ((tail != entry)
                ? tail
                : NULL);
      }
    }
   #endif
    template <class entry_t>
    bool ring_t<entry_t>::needsFree() const {
      // Object has no more references, safe to free now
      return useRefs && (head == NULL);
    }

    template <class entry_t>
    int ring_t<entry_t>::length() const {
      if (!head) {
        return 0;
      }

      ringEntry_t *ptr = head->rightRingEntry;
      int count = 1;
      while (ptr != head) {
        ++count;
        ptr = ptr->rightRingEntry;
      }
      return count;
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
     #if OCCA_THREAD_SHARABLE_ENABLED
      ring_t<entry_t>::mutex.lock();
     #endif
      typename entryRingMap_t::iterator it = rings.find(entry);
      if (it != rings.end()) {
        ring_t<entry_t> &ring = it->second;
        ring.removeRef(entry, false);
        rings.erase(it);
        // Change key if head changed
        if (ring.head &&
            ((entry_t*) ring.head != entry)) {
          rings[ring.head] = ring;
        }

      } else {
        entry->removeRef();
      }
     #if OCCA_THREAD_SHARABLE_ENABLED
      ring_t<entry_t>::mutex.unlock();
     #endif
    }

    template <class entry_t>
    bool multiRing_t<entry_t>::needsFree() const {
      return useRefs && rings.size();
    }
  }
}
