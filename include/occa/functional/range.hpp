#ifndef OCCA_FUNCTIONAL_RANGE_HEADER
#define OCCA_FUNCTIONAL_RANGE_HEADER

#include <occa/functional/array.hpp>

namespace occa {
  class range : public typelessArray {
  public:
    dim_t start;
    dim_t end;
    dim_t step;

    range(const dim_t end_);

    range(const dim_t start_,
          const dim_t end_);

    range(const dim_t start_,
          const dim_t end_,
          const dim_t step);

    range(occa::device device__,
          const dim_t end_);

    range(occa::device device__,
          const dim_t start_,
          const dim_t end_);

    range(occa::device device__,
          const dim_t start_,
          const dim_t end_,
          const dim_t step);

    range(const range &other);

    range& operator = (const range &other);

  private:
    void setupArrayScopeOverrides(occa::scope &scope) const;

    occa::scope getMapArrayScopeOverrides() const;
    occa::scope getReduceArrayScopeOverrides() const;

    std::string reductionInitialValue() const;

  public:
    udim_t length() const;

    //---[ Lambda methods ]-------------
    bool every(const occa::function<bool(const int)> &fn) const;

    bool some(const occa::function<bool(const int)> &fn) const;

    int findIndex(const occa::function<bool(const int)> &fn) const;

    void forEach(const occa::function<void(const int)> &fn) const;

    template <class T>
    array<T> map(const occa::function<T(const int)> &fn) const {
      return typelessMap<T>(fn);
    }

    template <class T>
    array<T> mapTo(occa::array<T> &output,
                    const occa::function<T(const int)> &fn) const {
      output.resize(length());
      typelessMapTo(output.memory(), fn);
      return output;
    }

    template <class T>
    T reduce(reductionType type,
              const occa::function<T(const T&, const int)> &fn) const {
      return typelessReduce<T>(type, T(), false, fn);
    }

    template <class T>
    T reduce(reductionType type,
              const T &localInit,
              const occa::function<T(const T&, const int)> &fn) const {
      return typelessReduce<T>(type, localInit, true, fn);
    }
    //==================================

    //---[ Utility methods ]------------
    array<int> toArray() const;
    //==================================
  };
}

#endif
