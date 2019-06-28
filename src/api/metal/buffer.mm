#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#include <occa/api/metal/buffer.hpp>

namespace occa {
  namespace api {
    namespace metal {
      buffer_t::buffer_t() {}

      buffer_t::buffer_t(const buffer_t &other) {}

      void buffer_t::free() {}

      void* buffer_t::getPtr() const {
        return (void*) 0;
      }
    }
  }
}

#endif
