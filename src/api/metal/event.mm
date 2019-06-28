#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#include <occa/api/metal/event.hpp>

namespace occa {
  namespace api {
    namespace metal {
      event_t::event_t() {}

      event_t::event_t(const event_t &other) {}

      void event_t::free() {}

      double event_t::getTime() const {
        return 0;
      }
    }
  }
}

#endif
