#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#include <occa/api/metal/commandQueue.hpp>

namespace occa {
  namespace api {
    namespace metal {
      commandQueue_t::commandQueue_t() {}

      commandQueue_t::commandQueue_t(const commandQueue_t &other) {}

      void commandQueue_t::free() {}
    }
  }
}

#endif
