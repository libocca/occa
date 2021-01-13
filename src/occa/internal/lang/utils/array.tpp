namespace occa {
  namespace lang {
    template <class TM>
    template <typename F>
    void array<TM>::rawFilter(F func, vectorType &returnVec) const {
      for (auto &entry : data) {
        if (func(entry)) {
          returnVec.push_back(entry);
        }
      }
    }

    template <class TM>
    template <class TM2, typename F>
    array<TM2> array<TM>::map(F func) const {
      array<TM2> newArray;

      for (auto &entry : data) {
        newArray.push(
          func(entry)
        );
      }

      return newArray;
    }

    template <class TM>
    template <class TM2, typename F>
    array<TM2> array<TM>::flatMap(F func) const {
      array<TM2> newArray;

      for (auto &entry : data) {
        for (auto &mappedEntry : func(entry)) {
          newArray.push(mappedEntry);
        }
      }

      return newArray;
    }

    template <class TM>
    template <typename F>
    void array<TM>::forEach(F func) const {
      for (auto &entry : data) {
        func(entry);
      }
    }

    template <class TM>
    bool array<TM>::startsWith(const array &other) const {
      const size_t otherLength = other.length();

      if (otherLength > length()) {
        return false;
      }

      for (size_t i = 0; i < otherLength; ++i) {
        if (other[i] != data[i]) {
          return false;
        }
      }

      return true;
    }
  }
}
