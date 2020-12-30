#ifndef OCCA_INTERNAL_API_METAL_BUFFER_HEADER
#define OCCA_INTERNAL_API_METAL_BUFFER_HEADER

#include <cstddef>

namespace occa {
  namespace api {
    namespace metal {
      class buffer_t {
       public:
        void *bufferObj;
        mutable void *ptr;

        buffer_t(void *obj_ = NULL);
        buffer_t(const buffer_t &other);

        buffer_t& operator = (const buffer_t &other);

        void free();

        void* getPtr() const;
      };
    }
  }
}

#endif
