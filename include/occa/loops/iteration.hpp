#ifndef OCCA_LOOPS_ITERATION_HEADER
#define OCCA_LOOPS_ITERATION_HEADER

#include <occa/functional/array.hpp>
#include <occa/functional/range.hpp>

namespace occa {
  enum class iterationType {
    undefined,
    range,
    indexArray
  };

  class iteration {
  private:
    iterationType type;
    occa::range range;
    occa::array<int> indices;

  public:
    iteration();

    iteration(const int rangeEnd);

    iteration(const occa::range &range_);

    iteration(const occa::array<int> &indices_);

    iteration(const iteration &other);

    iteration& operator = (const iteration &other);

    std::string buildForLoop(occa::scope &scope,
                             const std::string &iteratorName) const;

    std::string buildRangeForLoop(occa::scope &scope,
                                  const std::string &iteratorName) const;

    std::string buildIndexForLoop(occa::scope &scope,
                                  const std::string &iteratorName) const;
  };
}

#endif
