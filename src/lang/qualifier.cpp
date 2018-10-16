#include <sstream>

#include <occa/defines.hpp>
#include <occa/tools/sys.hpp>

#include <occa/lang/type.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/expression.hpp>
#include <occa/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace qualifierType {
      const udim_t none          = (1ULL << 0);

      const udim_t auto_         = (1ULL << 1);
      const udim_t const_        = (1ULL << 2);
      const udim_t constexpr_    = (1ULL << 3);
      const udim_t signed_       = (1ULL << 4);
      const udim_t unsigned_     = (1ULL << 5);
      const udim_t volatile_     = (1ULL << 6);
      const udim_t long_         = (1ULL << 7);
      const udim_t longlong_     = (1ULL << 8);
      const udim_t register_     = (1ULL << 9);

      const udim_t typeInfo_     = (1ULL << 10);
      const udim_t typeInfo      = (const_     |
                                    constexpr_ |
                                    signed_    |
                                    unsigned_  |
                                    volatile_  |
                                    long_      |
                                    longlong_  |
                                    register_  |
                                    typeInfo_);

      const udim_t forPointers_  = (1ULL << 11);
      const udim_t forPointers   = (const_    |
                                    volatile_ |
                                    forPointers_);

      const udim_t extern_       = (1ULL << 12);
      const udim_t externC       = (1ULL << 13);
      const udim_t externCpp     = (1ULL << 14);
      const udim_t static_       = (1ULL << 15);
      const udim_t thread_local_ = (1ULL << 16);

      const udim_t globalScope_  = (1ULL << 17);
      const udim_t globalScope   = (extern_       |
                                    externC       |
                                    externCpp     |
                                    static_       |
                                    thread_local_ |
                                    globalScope_);

      const udim_t friend_       = (1ULL << 18);
      const udim_t mutable_      = (1ULL << 19);

      const udim_t classInfo_    = (1ULL << 20);
      const udim_t classInfo     = (friend_  |
                                    mutable_ |
                                    classInfo_);

      const udim_t inline_       = (1ULL << 21);
      const udim_t virtual_      = (1ULL << 22);
      const udim_t explicit_     = (1ULL << 23);

      const udim_t functionInfo_ = (1ULL << 24);
      const udim_t functionInfo  = (typeInfo  |
                                    inline_   |
                                    virtual_  |
                                    explicit_ |
                                    functionInfo_);

      const udim_t builtin_      = (1ULL << 25);
      const udim_t typedef_      = (1ULL << 26);
      const udim_t class_        = (1ULL << 27);
      const udim_t enum_         = (1ULL << 28);
      const udim_t struct_       = (1ULL << 29);
      const udim_t union_        = (1ULL << 30);

      // Windows types
      const udim_t dllexport_    = (1ULL << 31);

      const udim_t newType_      = (1ULL << 32);
      const udim_t newType       = (typedef_ |
                                    class_   |
                                    enum_    |
                                    struct_  |
                                    union_   |
                                    newType_);

      const udim_t custom        = (1ULL << 33);
    }

    //---[ Qualifier ]------------------
    qualifier_t::qualifier_t(const std::string &name_,
                             const udim_t qtype_) :
      name(name_),
      qtype(qtype_) {}

    qualifier_t::~qualifier_t() {}

    udim_t qualifier_t::type() const {
      return qtype;
    }

    printer& operator << (printer &pout,
                          const qualifier_t &qualifier) {
      pout << qualifier.name;
      return pout;
    }
    //==================================

    //---[ Qualifiers ]-----------------
    qualifierWithSource::qualifierWithSource(const qualifier_t &qualifier_) :
      origin(),
      qualifier(&qualifier_) {}

    qualifierWithSource::qualifierWithSource(const fileOrigin &origin_,
                                             const qualifier_t &qualifier_) :
      origin(origin_),
      qualifier(&qualifier_) {}

    void qualifierWithSource::printWarning(const std::string &message) const {
      origin.printWarning(message);
    }
    void qualifierWithSource::printError(const std::string &message) const {
      origin.printError(message);
    }

    qualifiers_t::qualifiers_t() {}

    qualifiers_t::~qualifiers_t() {}

    void qualifiers_t::clear() {
      qualifiers.clear();
    }

    const qualifier_t* qualifiers_t::operator [] (const int index) const {
      if ((index < 0) ||
          (index >= (int) qualifiers.size())) {
        return NULL;
      }
      return qualifiers[index].qualifier;
    }

    int qualifiers_t::indexOf(const qualifier_t &qualifier) const {
      const int count = (int) qualifiers.size();
      if (count) {
        for (int i = 0; i < count; ++i) {
          if (qualifiers[i].qualifier == &qualifier) {
            return i;
          }
        }
      }
      return -1;
    }

    bool qualifiers_t::has(const qualifier_t &qualifier) const {
      return (indexOf(qualifier) >= 0);
    }

    bool qualifiers_t::operator == (const qualifiers_t &other) const {
      const int count      = (int) qualifiers.size();
      const int otherCount = (int) other.qualifiers.size();
      if (count != otherCount) {
        return false;
      }
      for (int i = 0; i < count; ++i) {
        if (!other.has(*(qualifiers[i].qualifier))) {
          return false;
        }
      }
      return true;
    }

    bool qualifiers_t::operator != (const qualifiers_t &other) const {
      return !((*this) == other);
    }

    qualifiers_t& qualifiers_t::operator += (const qualifier_t &qualifier) {
      if (!has(qualifier)) {
        qualifiers.push_back(qualifier);
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::operator -= (const qualifier_t &qualifier) {
      const int idx = indexOf(qualifier);
      if (idx >= 0) {
        qualifiers.erase(qualifiers.begin() + idx);
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::operator += (const qualifiers_t &other) {
      const int count = (int) other.qualifiers.size();
      for (int i = 0; i < count; ++i) {
        this->add(other.qualifiers[i]);
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::add(const fileOrigin &origin,
                                    const qualifier_t &qualifier) {
      if (!has(qualifier)) {
        qualifiers.push_back(
          qualifierWithSource(origin, qualifier)
        );
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::add(const qualifierWithSource &qualifier) {
      if (!has(*(qualifier.qualifier))) {
        qualifiers.push_back(qualifier);
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::add(const int index,
                                    const fileOrigin &origin,
                                    const qualifier_t &qualifier) {
      if (!has(qualifier)) {
        int safeIndex = 0;
        if (index > (int) qualifiers.size()) {
          safeIndex = (int) qualifiers.size();
        }
        qualifiers.insert(
          qualifiers.begin() + safeIndex,
          qualifierWithSource(origin, qualifier)
        );
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::add(const int index,
                                    const qualifierWithSource &qualifier) {
      if (!has(*(qualifier.qualifier))) {
        int safeIndex = 0;
        if (index > (int) qualifiers.size()) {
          safeIndex = (int) qualifiers.size();
        }
        qualifiers.insert(
          qualifiers.begin() + safeIndex,
          qualifier
        );
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::addFirst(const fileOrigin &origin,
                                         const qualifier_t &qualifier) {
      return addFirst(
        qualifierWithSource(origin, qualifier)
      );
    }

    qualifiers_t& qualifiers_t::addFirst(const qualifierWithSource &qualifier) {
      if (has(*(qualifier.qualifier))) {
        return *this;
      }
      const int count = (int) qualifiers.size();
      if (!count) {
        qualifiers.push_back(qualifier);
        return *this;
      }

      qualifiers.push_back(qualifiers[count - 1]);
      for (int i = 0; i < (count - 1); ++i) {
        qualifiers[i + 1] = qualifiers[i];
      }
      qualifiers[0] = qualifier;
      return *this;
    }

    printer& operator << (printer &pout,
                          const qualifiers_t &qualifiers) {
      const qualifierVector_t &quals = qualifiers.qualifiers;

      const int count = (int) quals.size();
      if (!count) {
        return pout;
      }
      pout << *(quals[0].qualifier);
      for (int i = 1; i < count; ++i) {
        pout << ' ' << *(quals[i].qualifier);
      }
      return pout;
    }
    //==================================
  }
}
