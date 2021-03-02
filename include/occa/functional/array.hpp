#ifndef OCCA_FUNCTIONAL_ARRAY_HEADER
#define OCCA_FUNCTIONAL_ARRAY_HEADER

#include <occa/functional/typelessArray.hpp>

namespace occa {
  class kernelArg;

  template <class T>
  class array : public typelessArray {
    template <class T2>
    friend class array;

  private:
    occa::memory memory_;

  public:
    array() :
      typelessArray() {}

    array(const dim_t size) :
      typelessArray() {
      resize(occa::getDevice(), size);
    }

    array(occa::device device, const dim_t size) :
      typelessArray() {
      resize(device, size);
    }

    array(occa::memory mem) :
      typelessArray(),
      memory_(mem) {

      memory_.setDtype(dtype::get<T>());
      setupTypelessArray(memory_);
    }
    array(const array<T> &other) :
      typelessArray(other),
      memory_(other.memory_) {}

    array& operator = (const array<T> &other) {
      typelessArray::operator = (other);
      memory_ = other.memory_;

      return *this;
    }

    occa::scope getMapArrayScopeOverrides() const {
      return occa::scope({
        {"occa_array_ptr", memory_}
      }, {
        {"defines/OCCA_ARRAY_FUNCTION_CALL(INDEX)",
         "OCCA_ARRAY_FUNCTION(occa_array_ptr[INDEX], INDEX, occa_array_ptr)"}
      });
    }

    occa::scope getReduceArrayScopeOverrides() const {
      return occa::scope({
        {"occa_array_ptr", memory_}
      }, {
        {"defines/OCCA_ARRAY_FUNCTION_CALL(ACC, INDEX)",
         "OCCA_ARRAY_FUNCTION(ACC, occa_array_ptr[INDEX], INDEX, occa_array_ptr)"}
      });
    }

    std::string reductionInitialValue() const {
      return "occa_array_ptr[0]";
    }

  public:
    //---[ Memory methods ]-------------
    bool isInitialized() const {
      return memory_.isInitialized();
    }

    occa::memory memory() const {
      return memory_;
    }

    operator occa::memory () const {
      return memory_;
    }

    operator kernelArg() {
      return memory_;
    }

    operator kernelArg() const {
      return memory_;
    }

    void resize(const dim_t size) {
      resize(device_, size);
    }

    void resize(occa::device device, const dim_t size) {
      if (size == (dim_t) length()) {
        return;
      }

      occa::memory prevMemory = memory_;
      memory_ = device.malloc<T>(size);

      if (prevMemory.isInitialized()) {
        if (prevMemory.length() < memory_.length()) {
          prevMemory.copyTo(memory_);
        } else {
          memory_.copyFrom(prevMemory);
        }
      }

      setupTypelessArray(memory_);
    }

    udim_t length() const {
      return memory_.length();
    }

    array clone() const {
      return array(memory_.clone());
    }

    void copyFrom(const T *src,
                  const dim_t entries = -1) {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyFrom(src, safeEntries * sizeof(T));
    }

  void copyFrom(const occa::memory src,
                const dim_t entries = -1) {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyFrom(src, safeEntries * sizeof(T));
    }

    void copyTo(T *dest,
                const dim_t entries = -1) const {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyTo(dest, safeEntries * sizeof(T));
    }

    void copyTo(occa::memory dest,
                const dim_t entries = -1) const {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyTo(dest, safeEntries * sizeof(T));
    }
    //==================================

    //---[ Lambda methods ]-------------
  public:
    bool every(const occa::function<bool(const T&)> &fn) const {
      return typelessEvery(fn);
    }

    bool every(const occa::function<bool(const T&, const int)> &fn) const {
      return typelessEvery(fn);
    }

    bool every(const occa::function<bool(const T&, const int, const T*)> &fn) const {
      return typelessEvery(fn);
    }

    bool some(const occa::function<bool(const T&)> &fn) const {
      return typelessSome(fn);
    }

    bool some(const occa::function<bool(const T&, const int)> &fn) const {
      return typelessSome(fn);
    }

    bool some(const occa::function<bool(const T&, const int, const T*)> &fn) const {
      return typelessSome(fn);
    }

    int findIndex(const occa::function<bool(const T&)> &fn) const {
      return typelessFindIndex(fn);
    }

    int findIndex(const occa::function<bool(const T&, const int)> &fn) const {
      return typelessFindIndex(fn);
    }

    int findIndex(const occa::function<bool(const T&, const int, const T*)> &fn) const {
      return typelessFindIndex(fn);
    }

    void forEach(const occa::function<void(const T&)> &fn) const {
      return typelessForEach(fn);
    }

    void forEach(const occa::function<void(const T&, const int)> &fn) const {
      return typelessForEach(fn);
    }

    void forEach(const occa::function<void(const T&, const int, const T*)> &fn) const {
      return typelessForEach(fn);
    }

    template <class T2>
    array<T2> map(const occa::function<T2(const T&)> &fn) const {
      return typelessMap<T2>(fn);
    }

    template <class T2>
    array<T2> map(const occa::function<T2(const T&, const int)> &fn) const {
      return typelessMap<T2>(fn);
    }

    template <class T2>
    array<T2> map(const occa::function<T2(const T&, const int, const T*)> &fn) const {
      return typelessMap<T2>(fn);
    }

    template <class T2>
    array<T2> mapTo(occa::array<T2> &output,
                     const occa::function<T2(const T&)> &fn) const {
      output.resize(length());
      typelessMapTo(output.memory_, fn);
      return output;
    }

    template <class T2>
    array<T2> mapTo(occa::array<T2> &output,
                     const occa::function<T2(const T&, const int)> &fn) const {
      output.resize(length());
      typelessMapTo(output.memory_, fn);
      return output;
    }

    template <class T2>
    array<T2> mapTo(occa::array<T2> &output,
                     const occa::function<T2(const T&, const int, const T*)> &fn) const {
      typelessMapTo(output.memory_, fn);
      return output;
    }

    template <class T2>
    T2 reduce(reductionType type,
               const occa::function<T2(const T2&, const T&)> &fn) const {
      return typelessReduce<T2>(type, T2(), false, fn);
    }

    template <class T2>
    T2 reduce(reductionType type,
               const occa::function<T2(const T2&, const T&, const int)> &fn) const {
      return typelessReduce<T2>(type, T2(), false, fn);
    }

    template <class T2>
    T2 reduce(reductionType type,
               occa::function<T2(const T2&, const T&, const int, const T*)> fn) const {
      return typelessReduce<T2>(type, T2(), false, fn);
    }

    template <class T2>
    T2 reduce(reductionType type,
               const T2 &localInit,
               const occa::function<T2(const T2&, const T&)> &fn) const {
      return typelessReduce<T2>(type, localInit, true, fn);
    }

    template <class T2>
    T2 reduce(reductionType type,
               const T2 &localInit,
               const occa::function<T2(const T2&, const T&, const int)> &fn) const {
      return typelessReduce<T2>(type, localInit, true, fn);
    }

    template <class T2>
    T2 reduce(reductionType type,
               const T2 &localInit,
               occa::function<T2(const T2&, const T&, const int, const T*)> fn) const {
      return typelessReduce<T2>(type, localInit, true, fn);
    }
    //==================================

    //---[ Utility methods ]------------
    T& operator [] (const dim_t index) {
      static T value;
      memory_.copyTo(&value,
                     sizeof(T),
                     index * sizeof(T));
      return value;
    }

    T& operator [] (const dim_t index) const {
      static T value;
      memory_.copyTo(&value,
                     sizeof(T),
                     index * sizeof(T));
      return value;
    }

    array slice(const dim_t offset,
                const dim_t count = -1) const {
      return array(
        memory_.slice(offset, count)
      );
    }

    array concat(const array &other) const {
      const udim_t bytes1 = memory_.size();
      const udim_t bytes2 = other.memory_.size();

      occa::memory ret = getDevice().template malloc<T>(length() + other.length());
      ret.copyFrom(memory_, bytes1, 0);
      ret.copyFrom(other.memory_, bytes2, bytes1);

      return array(ret);
    }

    array fill(const T &fillValue) {
      occa::scope fnScope({
        {"fillValue", fillValue}
      });

      return mapTo<T>(
        *this,
        OCCA_FUNCTION(fnScope, [=](const T &value) -> T {
          return fillValue;
        })
      );
    }

    bool includes(const T &target) const {
      occa::scope fnScope({
        {"target", target}
      });

      return some(
        OCCA_FUNCTION(fnScope, [=](const T &value) -> bool {
          return target == value;
        })
      );
    }

    dim_t indexOf(const T &target) const {
      occa::scope fnScope({
        {"target", target}
      });

      const int _length = (int) length();

      const int returnValue = reduce<int>(
        reductionType::min,
        (int) _length,
        OCCA_FUNCTION(fnScope, [=](const int &foundIndex, const T &value,const  int index) -> int {
          if ((target != value) || (foundIndex <= index)) {
            return foundIndex;
          }
          return index;
        })
      );

      return returnValue < _length ? returnValue : -1;
    }

    dim_t lastIndexOf(const T &target) const {
      occa::scope fnScope({
        {"target", target}
      });

      return reduce<int>(
        reductionType::max,
        -1,
        OCCA_FUNCTION(fnScope, [=](const int &foundIndex, const T &value, const int index) -> int {
          if ((target != value) || (foundIndex >= index)) {
            return foundIndex;
          }
          return index;
        })
      );
    }

    template <class T2>
    array<T2> cast() const {
      occa::scope fnScope({}, {
        {"defines/T2", dtype::get<T>().name()}
      });

      return map<T2>(
        OCCA_FUNCTION(fnScope, [=](const T &value) -> T2 {
          return (T2) value;
        })
      );
    }

    array reverse() const {
      const int size = (int) length();

      occa::scope fnScope({
        {"size", size}
      });

      return map<T>(
        OCCA_FUNCTION(fnScope, [=](const T &value, const int index, const T *values) -> T {
          return values[size - index - 1];
        })
      );
    }

    array shiftLeft(const int offset,
                    const T emptyValue = T()) const {
      if (offset == 0) {
        return clone();
      }

      const int size = (int) length();

      occa::scope fnScope({
        {"size", size},
        {"offset", offset},
        {"emptyValue", emptyValue},
      });

      return map<T>(
        OCCA_FUNCTION(fnScope, [=](const T &value, const int index, const T *values) -> T {
          if (index < (size - offset)) {
            return values[index + offset];
          } else {
            return emptyValue;
          }
        })
      );
    }

    array shiftRight(const int offset,
                     const T emptyValue = T()) const {
      if (offset == 0) {
        return clone();
      }

      occa::scope fnScope({
        {"size", (int) length()},
        {"offset", offset},
        {"emptyValue", emptyValue},
      });

      return map<T>(
        OCCA_FUNCTION(fnScope, [=](const T &value, const int index, const T *values) -> T {
          if (index >= offset) {
            return values[index - offset];
          } else {
            return emptyValue;
          }
        })
      );
    }

    T max() const {
      return reduce<T>(
        reductionType::max,
        OCCA_FUNCTION([=](const T &currentMax, const T &value) -> T {
          return currentMax > value ? currentMax : value;
        })
      );
    }

    T min() const {
      return reduce<T>(
        reductionType::min,
        OCCA_FUNCTION([=](const T &currentMin, const T &value) -> T {
          return currentMin < value ? currentMin : value;
        })
      );
    }
    //==================================

    //---[ Linear Algebra Methods ]-----
    T dotProduct(const array<T> &other) {
      occa::scope fnScope({
        {"other", other}
      });

      return reduce<T>(
        reductionType::sum,
        OCCA_FUNCTION(fnScope, [=](const T &acc, const T &value, const int index) -> T {
          return acc + (value * other[index]);
        })
      );
    }

    array clamp(const T minValue,
                const T maxValue) {
      occa::scope fnScope({
        {"minValue", minValue},
        {"maxValue", maxValue},
      });

      return map<T>(
        OCCA_FUNCTION(fnScope, [=](const T &value) -> T {
          const T valueWithMaxClamp = value > maxValue ? maxValue : value;
          return valueWithMaxClamp < minValue ? minValue : valueWithMaxClamp;
        })
      );
    }

    array clampMin(const T minValue) {
      occa::scope fnScope({
        {"minValue", minValue},
      });

      return map<T>(
        OCCA_FUNCTION(fnScope, [=](const T &value) -> T {
          return value < minValue ? minValue : value;
        })
      );
    }

    array clampMax(const T maxValue) {
      occa::scope fnScope({
        {"maxValue", maxValue},
      });

      return map<T>(
        OCCA_FUNCTION(fnScope, [=](const T &value) -> T {
          return value > maxValue ? maxValue : value;
        })
      );
    }
    //==================================
  };
}

#endif
