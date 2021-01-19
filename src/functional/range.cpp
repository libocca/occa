#include <occa/functional/range.hpp>

namespace occa {
  range::range(const dim_t end_) :
    range(occa::getDevice(),
          end_) {}

  range::range(const dim_t start_,
               const dim_t end_) :
    range(occa::getDevice(),
          start_,
          end_) {}

  range::range(const dim_t start_,
               const dim_t end_,
               const dim_t step_) :
    range(occa::getDevice(),
          start_,
          end_,
          step_) {}

  range::range(occa::device device__,
               const dim_t end_) :
    start(0),
    end(end_),
    step(end >= 0 ? 1 : -1) {

    setupTypelessArray(device__, dtype::get<int>());
  }

  range::range(occa::device device__,
               const dim_t start_,
               const dim_t end_) :
    start(start_),
    end(end_),
    step(end >= start ? 1 : -1) {

    setupTypelessArray(device__, dtype::get<int>());
  }

  range::range(occa::device device__,
               const dim_t start_,
               const dim_t end_,
               const dim_t step_) :
    start(start_),
    end(end_),
    step(step_ != 0 ? step_ : 1) {

    setupTypelessArray(device__, dtype::get<int>());
  }

  range::range(const range &other) :
    typelessArray(other),
    start(other.start),
    end(other.end),
    step(other.step) {}

  range& range::operator = (const range &other) {
    typelessArray::operator = (other);

    start = other.start;
    end = other.end;
    step = other.step;

    return *this;
  }

  void range::setupArrayScopeOverrides(occa::scope &scope) const {
    // Step compile-time defines on the common cases:
    // - Starting at 0
    // - Step of 1 or -1 (++/--)
    if (start) {
      scope.add("occa_range_start", start);
    } else {
      scope.props["defines/occa_range_start"] = 0;
    }

    if (step != 1 && step != -1) {
      scope.add("occa_range_step", step);
    } else {
      scope.props["defines/occa_range_step"] = step;
    }

    scope.add("occa_range_end", end);
  }

  occa::scope range::getMapArrayScopeOverrides() const {
    occa::scope scope;

    setupArrayScopeOverrides(scope);
    scope.props["defines/OCCA_ARRAY_FUNCTION_CALL(INDEX)"] = (
      "OCCA_ARRAY_FUNCTION(occa_range_start + (occa_range_step * INDEX), _, _)"
    );

    return scope;
  }

  occa::scope range::getReduceArrayScopeOverrides() const {
    occa::scope scope;

    setupArrayScopeOverrides(scope);
    scope.props["defines/OCCA_ARRAY_FUNCTION_CALL(ACC, INDEX)"] = (
      "OCCA_ARRAY_FUNCTION(ACC, occa_range_start + (occa_range_step * INDEX), _, _)"
    );

    return scope;
  }

  std::string range::reductionInitialValue() const {
    return "occa_range_start";
  }

  udim_t range::length() const {
    if (((start < end) && (step <= 0)) ||
        ((start > end) && (step >= 0))) {
      return 0;
    }

    return (
      step > 0
      ? (end - start + step - 1) / step
      : (end - start + step + 1) / step
    );
  }

  //---[ Lambda methods ]---------------
  bool range::every(const occa::function<bool(const int)> &fn) const {
    return typelessEvery(fn);
  }

  bool range::some(const occa::function<bool(const int)> &fn) const {
    return typelessSome(fn);
  }

  int range::findIndex(const occa::function<bool(const int)> &fn) const {
    return typelessFindIndex(fn);
  }

  void range::forEach(const occa::function<void(const int)> &fn) const {
    return typelessForEach(fn);
  }
  //====================================

  //---[ Utility methods ]--------------
  array<int> range::toArray() const {
    return map(OCCA_FUNCTION([=](const int index) -> int {
      return index;
    }));
  }
  //====================================
}
