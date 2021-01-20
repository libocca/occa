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

    template <class TM>
    array<TM> map(const occa::function<TM(const int)> &fn) const {
      return typelessMap<TM>(fn);
    }

    template <class TM>
    array<TM> mapTo(occa::array<TM> &output,
                    const occa::function<TM(const int)> &fn) const {
      output.resize(length());
      typelessMapTo(output.memory(), fn);
      return output;
    }

    template <class TM>
    TM reduce(reductionType type,
              const occa::function<TM(const TM&, const int)> &fn) const {
      return typelessReduce<TM>(type, TM(), false, fn);
    }

    template <class TM>
    TM reduce(reductionType type,
              const TM &localInit,
              const occa::function<TM(const TM&, const int)> &fn) const {
      return typelessReduce<TM>(type, localInit, true, fn);
    }
    //==================================

    //---[ Utility methods ]------------
    array<int> toArray() const;
    //==================================
  };
}

#endif
