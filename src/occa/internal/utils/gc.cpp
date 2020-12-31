#include <occa/internal/utils/gc.hpp>
#include <iostream>

namespace occa {
  namespace gc {
    withRefs::withRefs() :
      refs(0) {}

    int withRefs::getRefs() const {
      return refs;
    }

    void withRefs::addRef() {
      if (refs >= 0) {
        ++refs;
      }
    }

    int withRefs::removeRef() {
      if (refs > 0) {
        return --refs;
      }
      return refs;
    }

    void withRefs::setRefs(const int refs_) {
      refs = refs_;
    }

    void withRefs::dontUseRefs() {
      refs = -1;
    }

    ringEntry_t::ringEntry_t() :
      leftRingEntry(this),
      rightRingEntry(this) {}

    void ringEntry_t::removeRef() {
      if (leftRingEntry != this) {
        leftRingEntry->rightRingEntry = rightRingEntry;
        rightRingEntry->leftRingEntry = leftRingEntry;
      }
      leftRingEntry = rightRingEntry = this;
    }
  }
}
