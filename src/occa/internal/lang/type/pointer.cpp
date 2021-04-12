#include <occa/internal/lang/type/pointer.hpp>

namespace occa {
  namespace lang {
    pointer_t::pointer_t() {}

    pointer_t::pointer_t(const qualifiers_t &qualifiers_) :
      qualifiers(qualifiers_) {}

    pointer_t::pointer_t(const pointer_t &other) :
      qualifiers(other.qualifiers) {}

    pointer_t& pointer_t::operator = (const pointer_t &other) {
      qualifiers = other.qualifiers;
      return *this;
    }

    bool pointer_t::has(const qualifier_t &qualifier) const {
      return qualifiers.has(qualifier);
    }

    void pointer_t::operator += (const qualifier_t &qualifier) {
      qualifiers += qualifier;
    }

    void pointer_t::operator -= (const qualifier_t &qualifier) {
      qualifiers -= qualifier;
    }

    void pointer_t::operator += (const qualifiers_t &qualifiers_) {
      qualifiers += qualifiers_;
    }

    void pointer_t::add(const fileOrigin &origin,
                        const qualifier_t &qualifier) {
      qualifiers.add(origin, qualifier);
    }

    void pointer_t::add(const qualifierWithSource &qualifier) {
      qualifiers.add(qualifier);
    }

    io::output& operator << (io::output &out,
                               const pointer_t &pointer) {
      printer pout(out);
      pout << pointer;
      return out;
    }

    printer& operator << (printer &pout,
                          const pointer_t &pointer) {
      pout << '*';
      if (pointer.qualifiers.size()) {
        pout << ' ' << pointer.qualifiers;
      }
      return pout;
    }
  }
}
