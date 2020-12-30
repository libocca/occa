#ifndef OCCA_INTERNAL_LANG_UTILS_BASEARRAY_HEADER
#define OCCA_INTERNAL_LANG_UTILS_BASEARRAY_HEADER

#include <initializer_list>
#include <string>
#include <vector>

namespace occa {
  namespace lang {
    template <class TM>
    class array {
     public:
      typedef typename std::vector<TM> vectorType;
      typedef typename vectorType::iterator vectorIterator;
      typedef typename vectorType::const_iterator cVectorIterator;

      vectorType data;

      inline array(const vectorType &data_) :
          data(data_) {}

      inline array(const array &other) :
          data(other.data) {}

      inline array(std::initializer_list<TM> list) :
          data(list) {}

      // Implement for-loop iterators
      inline vectorIterator begin() noexcept {
        return data.begin();
      }

      inline cVectorIterator begin() const noexcept {
        return data.cbegin();
      }

      inline vectorIterator end() noexcept {
        return data.end();
      }

      inline cVectorIterator end() const noexcept {
        return data.cend();
      }

      // Basic array methods
      inline TM& operator [] (const size_t index) {
        return data[index];
      }

      inline const TM& operator [] (const size_t index) const {
        return data[index];
      }

      inline TM& last () {
        return data[data.size() - 1];
      }

      inline const TM& last () const {
        return data[data.size() - 1];
      }

      inline size_t length() const {
        return data.size();
      }

      inline bool isEmpty() const {
        return !data.size();
      }

      inline bool isNotEmpty() const {
        return data.size();
      }

      inline void push(const TM &value) {
        data.push_back(value);
      }

      inline TM pop() {
        TM lastValue = data.back();
        data.pop_back();
        return lastValue;
      }

      inline void insert(const size_t index, const TM &value) {
        data.insert(data.begin() + index, value);
      }

      inline void append(const array &other) {
        data.insert(data.end(), other.data.begin(), other.data.end());
      }

      inline void remove(const size_t index) {
        data.erase(data.begin() + index);
      }

      inline void clear() {
        data.clear();
      }

      inline void swap(array &other) {
        data.swap(other.data);
      }

      // Iterative methods
      template <typename F>
      void rawFilter(F func, vectorType &returnVec) const;

      template <class TM2, typename F>
      array<TM2> map(F func) const;

      template <class TM2, typename F>
      array<TM2> flatMap(F func) const;

      template <typename F>
      void forEach(F func) const;

      bool startsWith(const array &other) const;

      // Helper macro to generate functions that maintain derived-class return types
#define OCCA_LANG_ARRAY_DEFINE_BASE_METHODS(CLASS, CLASS_TM, TM)  \
  inline CLASS() {}                                               \
                                                                  \
  inline CLASS& operator = (const CLASS &other) {                 \
    data = other.data;                                            \
    return *this;                                                 \
  }                                                               \
                                                                  \
  static inline CLASS_TM from(vectorType &data_) {                \
    return CLASS(data_);                                          \
  }                                                               \
                                                                  \
  template <typename F>                                           \
  inline CLASS_TM filter(F func) const {                          \
    vectorType returnVec;                                         \
    rawFilter(func, returnVec);                                   \
    return returnVec;                                             \
  }                                                               \
                                                                  \
  inline CLASS_TM& inplaceReverse() {                             \
    const size_t size = data.size();                              \
    const size_t midPoint = size / 2;                             \
                                                                  \
    for (size_t i = 0; i < midPoint; ++i) {                       \
      TM entry = data[i];                                         \
      data[i] = data[size - i - 1];                               \
      data[size - i - 1] = entry;                                 \
    }                                                             \
                                                                  \
    return *this;                                                 \
  }                                                               \
                                                                  \
  inline CLASS_TM reverse() const {                               \
    CLASS_TM arr = *this;                                         \
    arr.inplaceReverse();                                         \
    return arr;                                                   \
  }                                                               \
                                                                  \
  inline CLASS_TM operator + (const CLASS_TM &other) const {      \
    CLASS_TM arr = *this;                                         \
    arr.data.insert(                                              \
      arr.data.end(),                                             \
      other.data.begin(),                                         \
      other.data.end()                                            \
    );                                                            \
    return arr;                                                   \
  }

#define OCCA_LANG_ARRAY_DEFINE_METHODS(CLASS, TM)                   \
      typedef array<TM> arrayType;                                  \
                                                                    \
      typedef typename std::vector<TM> vectorType;                  \
      typedef typename vectorType::iterator vectorIterator;         \
      typedef typename vectorType::const_iterator cVectorIterator;  \
                                                                    \
      inline CLASS(vectorType &data_) :                             \
          array(data_) {}                                           \
                                                                    \
      inline CLASS(const CLASS &other) :                            \
          array(other) {}                                           \
                                                                    \
      OCCA_LANG_ARRAY_DEFINE_BASE_METHODS(CLASS, CLASS, TM)

      OCCA_LANG_ARRAY_DEFINE_BASE_METHODS(array, array<TM>, TM)
    };
  }
}

#include "array.tpp"

#endif
