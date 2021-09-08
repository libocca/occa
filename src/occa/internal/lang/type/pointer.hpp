#ifndef OCCA_INTERNAL_LANG_TYPE_POINTER_HEADER
#define OCCA_INTERNAL_LANG_TYPE_POINTER_HEADER

#include <occa/internal/lang/type/type.hpp>

namespace occa {
  namespace lang {
    class pointer_t {
    public:
      qualifiers_t qualifiers;

      pointer_t() = default;
      explicit pointer_t(const qualifiers_t &qualifiers_);

      bool has(const qualifier_t &qualifier) const;

      void operator += (const qualifier_t &qualifier);
      void operator -= (const qualifier_t &qualifier);
      void operator += (const qualifiers_t &others);

      void add(const fileOrigin &origin,
               const qualifier_t &qualifier);

      void add(const qualifierWithSource &qualifier);
    };

    io::output& operator << (io::output &out,
                             const pointer_t &pointer);

    printer& operator << (printer &pout,
                          const pointer_t &pointer);
  }
}

#endif
