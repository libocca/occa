#ifndef OCCA_FUNCTIONAL_RANGE_HEADER
#define OCCA_FUNCTIONAL_RANGE_HEADER

#include <occa/functional/array.hpp>

namespace occa {
  class range : public typelessArray {
  public:
    const dim_t start;
    const dim_t end;
    const dim_t step;

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

  private:
    void setupArrayScopeOverrides(occa::scope &scope) const;

    occa::scope getMapArrayScopeOverrides() const;
    occa::scope getReduceArrayScopeOverrides() const;

    std::string reductionInitialValue() const;

  public:
    udim_t length() const;

    //---[ Lambda methods ]-------------
    inline bool every(const occa::function<bool(int)> &fn) const {
      return typelessEvery(fn);
    }

    inline bool some(const occa::function<bool(int)> &fn) const {
      return typelessSome(fn);
    }

    inline int findIndex(const occa::function<bool(int)> &fn) const {
      return typelessFindIndex(fn);
    }

    inline void forEach(const occa::function<void(int)> &fn) const {
      return typelessForEach(fn);
    }

    template <class TM>
    array<TM> map(const occa::function<TM(int)> &fn) const {
      return typelessMap<TM>(fn);
    }

    template <class TM>
    array<TM> mapTo(occa::array<TM> &output,
                    const occa::function<TM(int)> &fn) const {
      output.resize(length());
      typelessMapTo(output.memory(), fn);
      return output;
    }

    template <class TM>
    TM reduce(reductionType type,
              const occa::function<TM(TM, int)> &fn) const {
      return typelessReduce<TM>(type, TM(), false, fn);
    }

    template <class TM>
    TM reduce(reductionType type,
              const TM &localInit,
              const occa::function<TM(TM, int)> &fn) const {
      return typelessReduce<TM>(type, localInit, true, fn);
    }
    //==================================
  };
}

#endif
