#include <occa/loops/iteration.hpp>

namespace occa {
  iteration::iteration() :
    type(iterationType::undefined),
    range(-1),
    tileSize(0) {}

  iteration::iteration(const int rangeEnd) :
    type(iterationType::range),
    range(rangeEnd),
    tileSize(0) {}

  iteration::iteration(const occa::range &range_) :
    type(iterationType::range),
    range(range_),
    tileSize(0) {}

  iteration::iteration(const occa::array<int> &indices_) :
    type(iterationType::indexArray),
    range(-1),
    indices(indices_),
    tileSize(0) {}

  iteration::iteration(const iteration &other) :
    type(other.type),
    range(other.range),
    indices(other.indices),
    tileSize(other.tileSize) {}

  iteration& iteration::operator = (const iteration &other) {
    type     = other.type;
    range    = other.range;
    indices  = other.indices;
    tileSize = other.tileSize;

    return *this;
  }

  std::string iteration::buildForLoop(forLoopType loopType,
                                      occa::scope &scope,
                                      const std::string &iteratorName) const {
    OCCA_ERROR("Iteration not defined",
               type != iterationType::undefined);

    std::string attribute;
    switch (loopType) {
      case forLoopType::outer: {
        if (tileSize) {
          OCCA_ERROR("Tile size cannot be 0",
                     tileSize != 0);
          attribute = "@tile(" + std::to_string(tileSize) + ", @outer, @inner)";
        } else {
          attribute = "@outer";
        }
        break;
      }
      case forLoopType::inner: {
        attribute = "@inner";
        break;
      }
    }

    if (type == iterationType::range) {
      return buildRangeForLoop(scope, iteratorName, attribute);
    } else {
      return buildIndexForLoop(scope, iteratorName, attribute);
    }
  }

  std::string iteration::buildRangeForLoop(occa::scope &scope,
                                           const std::string &iteratorName,
                                           const std::string &forAttribute) const {
    const std::string startName = iteratorName + "_start";
    const std::string endName   = iteratorName + "_end";
    const std::string stepName  = iteratorName + "_step";

    // Step compile-time defines on the common cases:
    // - Starting at 0
    // - Step of 1 or -1 (++/--)
    if (range.start) {
      scope.add(startName, range.start);
    } else {
      scope.props["defines"][startName] = 0;
    }

    if (range.step != 1 && range.step != -1) {
      scope.add(stepName, range.step);
    } else {
      scope.props["defines"][stepName] = range.step;
    }

    scope.add(endName, range.end);

    const char compOperator = (
      range.step > 0
      ? '<'
      : '>'
    );

    const char stepOperator = (
      range.step > 0
      ? '+'
      : '-'
    );

    std::stringstream ss;

    // for (int idx = 0; idx < N; idx += 1; @attr) {
    ss << "for (int " << iteratorName << " = " << startName << ";"
       << ' ' << iteratorName << ' ' << compOperator << ' ' << endName << ";"
       << ' ' << iteratorName << " " << stepOperator << "= " << stepName << ";"
       << " " << forAttribute << ") {";

    return ss.str();
  }

  std::string iteration::buildIndexForLoop(occa::scope &scope,
                                           const std::string &iteratorName,
                                           const std::string &forAttribute) const {
    const std::string iteratorIndexName = iteratorName + "_index";
    const std::string iteratorLengthName = iteratorName + "_length";
    const std::string iteratorPtrName = iteratorName + "_ptr";

    scope.add(iteratorLengthName, indices.length());
    scope.add(iteratorPtrName, indices);

    std::stringstream ss;

    // for (int i = 0; i < N; i += 1; @attr) {
    //   idx = idcPtr[i];
    ss << "for (int " << iteratorIndexName << " = 0;"
       << " " << iteratorIndexName << " < " << iteratorLengthName << ";"
       << " ++" << iteratorIndexName << ";"
       << " " << forAttribute << ") {"
       << "  const int " << iteratorName << " = " << iteratorPtrName << "[" << iteratorIndexName << "];";

    return ss.str();
  }

  tileIteration::tileIteration(occa::iteration it_,
                               const int tileSize) :
    it(it_) {
    it.tileSize = tileSize;
  }

  tileIteration::operator iteration () const {
    return it;
  }
}
