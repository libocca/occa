#include <occa/modes/dpcpp/streamTag.hpp>
#include <occa/modes/dpcpp/utils.hpp>

namespace occa {
  namespace dpcpp {
    streamTag::streamTag(modeDevice_t *modeDevice_) : modeStreamTag_t(modeDevice_),
      time(-1) {}

    streamTag::~streamTag() {
    }
  }
}
