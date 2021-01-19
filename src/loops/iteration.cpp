#include <occa/loops/iteration.hpp>

namespace occa {
  iteration::iteration() :
    type(iterationType::undefined),
    range(-1) {}

  iteration::iteration(const int rangeEnd) :
    type(iterationType::range),
    range(rangeEnd) {}

  iteration::iteration(const occa::range &range_) :
    type(iterationType::range),
    range(range_) {}

  iteration::iteration(const occa::array<int> &indices_) :
    type(iterationType::indexArray),
    range(-1),
    indices(indices_) {}

  iteration::iteration(const iteration &other) :
    type(other.type),
    range(other.range),
    indices(other.indices) {}

  iteration& iteration::operator = (const iteration &other) {
    type    = other.type;
    range   = other.range;
    indices = other.indices;

    return *this;
  }

  std::string iteration::buildForLoop(occa::scope &scope,
                                      const std::string &iteratorName) const {
    OCCA_ERROR("Iteration not defined",
               type != iterationType::undefined);

    if (type == iterationType::range) {
      return buildRangeForLoop(scope, iteratorName);
    } else {
      return buildIndexForLoop(scope, iteratorName);
    }
  }

  std::string iteration::buildRangeForLoop(occa::scope &scope,
                                           const std::string &iteratorName) const {
    std::stringstream ss;
    ss << "for (int " << iteratorName << " = 0;"
       << " " << iteratorName << " < 10;"
       << " ++" << iteratorName << ") {";
    return ss.str();
  }

  std::string iteration::buildIndexForLoop(occa::scope &scope,
                                           const std::string &iteratorName) const {
    std::stringstream ss;
    ss << "for (int " << iteratorName << " = 0;"
       << " " << iteratorName << " < 10;"
       << " ++" << iteratorName << ") {";
    return ss.str();
  }
}
