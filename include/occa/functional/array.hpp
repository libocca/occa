#ifndef OCCA_FUNCTIONAL_ARRAY_HEADER
#define OCCA_FUNCTIONAL_ARRAY_HEADER

#include <occa/functional/typelessArray.hpp>

namespace occa {
  class kernelArg;

  template <class TM>
  class array : public typelessArray {
    template <class TM2>
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

      memory_.setDtype(dtype::get<TM>());
      setupTypelessArray(memory_);
    }
    array(const array<TM> &other) :
      typelessArray(other),
      memory_(other.memory_) {}

    array& operator = (const array<TM> &other) {
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
      memory_ = device.malloc<TM>(size);

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

    void copyFrom(const TM *src,
                  const dim_t entries = -1) {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyFrom(src, safeEntries * sizeof(TM));
    }

  void copyFrom(const occa::memory src,
                const dim_t entries = -1) {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyFrom(src, safeEntries * sizeof(TM));
    }

    void copyTo(TM *dest,
                const dim_t entries = -1) const {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyTo(dest, safeEntries * sizeof(TM));
    }

    void copyTo(occa::memory dest,
                const dim_t entries = -1) const {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyTo(dest, safeEntries * sizeof(TM));
    }
    //==================================

    //---[ Lambda methods ]-------------
  public:
    bool every(const occa::function<bool(const TM&)> &fn) const {
      return typelessEvery(fn);
    }

    bool every(const occa::function<bool(const TM&, const int)> &fn) const {
      return typelessEvery(fn);
    }

    bool every(const occa::function<bool(const TM&, const int, const TM*)> &fn) const {
      return typelessEvery(fn);
    }

    bool some(const occa::function<bool(const TM&)> &fn) const {
      return typelessSome(fn);
    }

    bool some(const occa::function<bool(const TM&, const int)> &fn) const {
      return typelessSome(fn);
    }

    bool some(const occa::function<bool(const TM&, const int, const TM*)> &fn) const {
      return typelessSome(fn);
    }

    int findIndex(const occa::function<bool(const TM&)> &fn) const {
      return typelessFindIndex(fn);
    }

    int findIndex(const occa::function<bool(const TM&, const int)> &fn) const {
      return typelessFindIndex(fn);
    }

    int findIndex(const occa::function<bool(const TM&, const int, const TM*)> &fn) const {
      return typelessFindIndex(fn);
    }

    void forEach(const occa::function<void(const TM&)> &fn) const {
      return typelessForEach(fn);
    }

    void forEach(const occa::function<void(const TM&, const int)> &fn) const {
      return typelessForEach(fn);
    }

    void forEach(const occa::function<void(const TM&, const int, const TM*)> &fn) const {
      return typelessForEach(fn);
    }

    template <class TM2>
    array<TM2> map(const occa::function<TM2(const TM&)> &fn) const {
      return typelessMap<TM2>(fn);
    }

    template <class TM2>
    array<TM2> map(const occa::function<TM2(const TM&, const int)> &fn) const {
      return typelessMap<TM2>(fn);
    }

    template <class TM2>
    array<TM2> map(const occa::function<TM2(const TM&, const int, const TM*)> &fn) const {
      return typelessMap<TM2>(fn);
    }

    template <class TM2>
    array<TM2> mapTo(occa::array<TM2> &output,
                     const occa::function<TM2(const TM&)> &fn) const {
      output.resize(length());
      typelessMapTo(output.memory_, fn);
      return output;
    }

    template <class TM2>
    array<TM2> mapTo(occa::array<TM2> &output,
                     const occa::function<TM2(const TM&, const int)> &fn) const {
      output.resize(length());
      typelessMapTo(output.memory_, fn);
      return output;
    }

    template <class TM2>
    array<TM2> mapTo(occa::array<TM2> &output,
                     const occa::function<TM2(const TM&, const int, const TM*)> &fn) const {
      typelessMapTo(output.memory_, fn);
      return output;
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const occa::function<TM2(const TM2&, const TM&)> &fn) const {
      return typelessReduce<TM2>(type, TM2(), false, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const occa::function<TM2(const TM2&, const TM&, const int)> &fn) const {
      return typelessReduce<TM2>(type, TM2(), false, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               occa::function<TM2(const TM2&, const TM&, const int, const TM*)> fn) const {
      return typelessReduce<TM2>(type, TM2(), false, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const TM2 &localInit,
               const occa::function<TM2(const TM2&, const TM&)> &fn) const {
      return typelessReduce<TM2>(type, localInit, true, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const TM2 &localInit,
               const occa::function<TM2(const TM2&, const TM&, const int)> &fn) const {
      return typelessReduce<TM2>(type, localInit, true, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const TM2 &localInit,
               occa::function<TM2(const TM2&, const TM&, const int, const TM*)> fn) const {
      return typelessReduce<TM2>(type, localInit, true, fn);
    }
    //==================================

    //---[ Utility methods ]------------
    TM& operator [] (const dim_t index) {
      static TM value;
      memory_.copyTo(&value,
                     sizeof(TM),
                     index * sizeof(TM));
      return value;
    }

    TM& operator [] (const dim_t index) const {
      static TM value;
      memory_.copyTo(&value,
                     sizeof(TM),
                     index * sizeof(TM));
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

      occa::memory ret = getDevice().template malloc<TM>(length() + other.length());
      ret.copyFrom(memory_, bytes1, 0);
      ret.copyFrom(other.memory_, bytes2, bytes1);

      return array(ret);
    }

    array fill(const TM &fillValue) {
      occa::scope fnScope({
        {"fillValue", fillValue}
      });

      return mapTo<TM>(
        *this,
        OCCA_FUNCTION(fnScope, [=](const TM &value) -> TM {
          return fillValue;
        })
      );
    }

    bool includes(const TM &target) const {
      occa::scope fnScope({
        {"target", target}
      });

      return some(
        OCCA_FUNCTION(fnScope, [=](const TM &value) -> bool {
          return target == value;
        })
      );
    }

    dim_t indexOf(const TM &target) const {
      occa::scope fnScope({
        {"target", target}
      });

      const int _length = (int) length();

      const int returnValue = reduce<int>(
        reductionType::min,
        (int) _length,
        OCCA_FUNCTION(fnScope, [=](const int &foundIndex, const TM &value,const  int index) -> int {
          if ((target != value) || (foundIndex <= index)) {
            return foundIndex;
          }
          return index;
        })
      );

      return returnValue < _length ? returnValue : -1;
    }

    dim_t lastIndexOf(const TM &target) const {
      occa::scope fnScope({
        {"target", target}
      });

      return reduce<int>(
        reductionType::max,
        -1,
        OCCA_FUNCTION(fnScope, [=](const int &foundIndex, const TM &value, const int index) -> int {
          if ((target != value) || (foundIndex >= index)) {
            return foundIndex;
          }
          return index;
        })
      );
    }

    template <class TM2>
    array<TM2> cast() const {
      occa::scope fnScope({}, {
        {"defines/TM2", dtype::get<TM>().name()}
      });

      return map<TM2>(
        OCCA_FUNCTION(fnScope, [=](const TM &value) -> TM2 {
          return (TM2) value;
        })
      );
    }

    array reverse() const {
      const int size = (int) length();

      occa::scope fnScope({
        {"size", size}
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](const TM &value, const int index, const TM *values) -> TM {
          return values[size - index - 1];
        })
      );
    }

    array shiftLeft(const int offset,
                    const TM emptyValue = TM()) const {
      if (offset == 0) {
        return clone();
      }

      const int size = (int) length();

      occa::scope fnScope({
        {"size", size},
        {"offset", offset},
        {"emptyValue", emptyValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](const TM &value, const int index, const TM *values) -> TM {
          if (index < (size - offset)) {
            return values[index + offset];
          } else {
            return emptyValue;
          }
        })
      );
    }

    array shiftRight(const int offset,
                     const TM emptyValue = TM()) const {
      if (offset == 0) {
        return clone();
      }

      const int size = (int) length();

      occa::scope fnScope({
        {"size", size},
        {"offset", offset},
        {"emptyValue", emptyValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](const TM &value, const int index, const TM *values) -> TM {
          if (index >= offset) {
            return values[index - offset];
          } else {
            return emptyValue;
          }
        })
      );
    }

    TM max() const {
      return reduce<TM>(
        reductionType::max,
        OCCA_FUNCTION([=](const TM &currentMax, const TM &value) -> TM {
          return currentMax > value ? currentMax : value;
        })
      );
    }

    TM min() const {
      return reduce<TM>(
        reductionType::min,
        OCCA_FUNCTION([=](const TM &currentMin, const TM &value) -> TM {
          return currentMin < value ? currentMin : value;
        })
      );
    }
    //==================================

    //---[ Linear Algebra Methods ]-----
    TM dotProduct(const array<TM> &other) {
      occa::scope fnScope({
        {"other", other}
      });

      return reduce<TM>(
        reductionType::sum,
        OCCA_FUNCTION(fnScope, [=](const TM &acc, const TM &value, const int index) -> TM {
          return acc + (value * other[index]);
        })
      );
    }

    array clamp(const TM minValue,
                const TM maxValue) {
      occa::scope fnScope({
        {"minValue", minValue},
        {"maxValue", maxValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](const TM &value) -> TM {
          const TM valueWithMaxClamp = value > maxValue ? maxValue : value;
          return valueWithMaxClamp < minValue ? minValue : valueWithMaxClamp;
        })
      );
    }

    array clampMin(const TM minValue) {
      occa::scope fnScope({
        {"minValue", minValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](const TM &value) -> TM {
          return value < minValue ? minValue : value;
        })
      );
    }

    array clampMax(const TM maxValue) {
      occa::scope fnScope({
        {"maxValue", maxValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](const TM &value) -> TM {
          return value > maxValue ? maxValue : value;
        })
      );
    }
    //==================================
  };
}

#endif
