#ifndef OCCA_API_METAL_BUFFER_HEADER
#define OCCA_API_METAL_BUFFER_HEADER

namespace occa {
  namespace api {
    namespace metal {
      class buffer_t {
       public:
        void *obj;

        buffer_t();
        buffer_t(const buffer_t &other);

        void free();

        void* getPtr() const;
      };
    }
  }
}

#endif
