#ifndef OCCA_INTERNAL_LANG_QUALIFIER_HEADER
#define OCCA_INTERNAL_LANG_QUALIFIER_HEADER

#include <vector>

#include <occa/types/primitive.hpp>
#include <occa/internal/lang/file.hpp>
#include <occa/internal/lang/printer.hpp>

namespace occa {
  namespace lang {
    namespace qualifierType {
      extern const udim_t none;

      extern const udim_t auto_;
      extern const udim_t const_;
      extern const udim_t constexpr_;
      extern const udim_t signed_;
      extern const udim_t unsigned_;
      extern const udim_t volatile_;
      extern const udim_t register_;
      extern const udim_t long_;
      extern const udim_t longlong_;
      extern const udim_t typeInfo;

      extern const udim_t forPointers_;
      extern const udim_t forPointers;

      extern const udim_t extern_;
      extern const udim_t externC;
      extern const udim_t externCpp;
      extern const udim_t static_;
      extern const udim_t thread_local_;

      extern const udim_t globalScope_;
      extern const udim_t globalScope;

      extern const udim_t friend_;
      extern const udim_t mutable_;

      extern const udim_t classInfo_;
      extern const udim_t classInfo;

      extern const udim_t inline_;
      extern const udim_t virtual_;
      extern const udim_t explicit_;

      extern const udim_t functionInfo_;
      extern const udim_t functionInfo;

      extern const udim_t builtin_;
      extern const udim_t typedef_;
      extern const udim_t class_;
      extern const udim_t enum_;
      extern const udim_t struct_;
      extern const udim_t union_;

      // Windows types
      extern const udim_t dllexport_;

      extern const udim_t newType_;
      extern const udim_t newType;

      extern const udim_t custom;
    }

    //---[ Qualifier ]------------------
    class qualifier_t {
    public:
      std::string name;
      const udim_t qtype;

      qualifier_t(const std::string &name_,
                  const udim_t qtype_);
      ~qualifier_t();

      bool operator == (const qualifier_t &other) const;

      udim_t type() const;
    };

    printer& operator << (printer &pout,
                          const qualifier_t &qualifier);
    //==================================

    //---[ Qualifiers ]-----------------
    class qualifierWithSource {
    public:
      fileOrigin origin;
      const qualifier_t *qualifier;

      qualifierWithSource(const qualifier_t &qualifier_);

      qualifierWithSource(const fileOrigin &origin_,
                          const qualifier_t &qualifier_);

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };

    typedef std::vector<qualifierWithSource> qualifierVector_t;

    class qualifiers_t {
    public:
      qualifierVector_t qualifiers;

      qualifiers_t();
      ~qualifiers_t();

      void clear();

      inline int size() const {
        return (int) qualifiers.size();
      }

      const qualifier_t* operator [] (const int index) const;

      int indexOf(const qualifier_t &qualifier) const;
      bool has(const qualifier_t &qualifier) const;

      bool operator == (const qualifiers_t &other) const;
      bool operator != (const qualifiers_t &other) const;

      qualifiers_t& operator += (const qualifier_t &qualifier);
      qualifiers_t& operator -= (const qualifier_t &qualifier);
      qualifiers_t& operator += (const qualifiers_t &others);

      qualifiers_t& add(const fileOrigin &origin,
                        const qualifier_t &qualifier);

      qualifiers_t& add(const qualifierWithSource &qualifier);

      qualifiers_t& add(const int index,
                        const fileOrigin &origin,
                        const qualifier_t &qualifier);

      qualifiers_t& add(const int index,
                        const qualifierWithSource &qualifier);

      qualifiers_t& addFirst(const fileOrigin &origin,
                             const qualifier_t &qualifier);

      qualifiers_t& addFirst(const qualifierWithSource &qualifier);

      void swap(qualifiers_t &other);
    };

    printer& operator << (printer &pout,
                          const qualifiers_t &qualifiers);
    //==================================
  }
}
#endif
