#ifndef OCCA_INTERNAL_LANG_TYPE_VARTYPE_HEADER
#define OCCA_INTERNAL_LANG_TYPE_VARTYPE_HEADER

#include <occa/internal/lang/type/type.hpp>

namespace occa {
  namespace lang {
    enum vartypePrintType_t {
      none,
      type,
      typeDeclaration
    };

    class vartype_t {
    public:
      qualifiers_t qualifiers;

      identifierToken *typeToken;
      const type_t *type;

      pointerVector pointers;
      token_t *referenceToken;
      arrayVector arrays;

      int bitfield;

      std::string customPrefix, customSuffix;

      vartype_t();

      vartype_t(const type_t &type_);

      vartype_t(const identifierToken &typeToken_,
                const type_t &type_);

      vartype_t(const vartype_t &other);

      ~vartype_t();

      vartype_t& operator = (const vartype_t &other);

      void setType(const identifierToken &typeToken_,
                   const type_t &type_);

      void setType(const type_t &type_);

      void clear();

      bool isValid() const;
      bool isNamed() const;
      std::string name() const;

      bool isUniqueType(const type_t *type_) const;

      fileOrigin origin() const;

      bool isPointerType() const;

      void setReferenceToken(token_t *token);
      bool isReference() const;

      dtype_t dtype() const;

      bool operator == (const vartype_t &other) const;
      bool operator != (const vartype_t &other) const;

      bool has(const qualifier_t &qualifier) const;

      vartype_t& operator += (const qualifier_t &qualifier);
      vartype_t& operator -= (const qualifier_t &qualifier);
      vartype_t& operator += (const qualifiers_t &qualifiers_);

      void add(const fileOrigin &origin,
               const qualifier_t &qualifier);

      void add(const qualifierWithSource &qualifier);

      void add(const int index,
               const fileOrigin &origin,
               const qualifier_t &qualifier);

      void add(const int index,
               const qualifierWithSource &qualifier);

      void remove(const qualifier_t &qualifier);

      vartype_t& operator += (const pointer_t &pointer);
      vartype_t& operator += (const pointerVector &pointers_);

      vartype_t& operator += (const array_t &array);
      vartype_t& operator += (const arrayVector &arrays_);

      bool hasAttribute(const std::string &attr) const;

      vartype_t declarationType() const;

      vartype_t flatten() const;

      bool definesStruct() const;

      void printDeclaration(printer &pout,
                            const std::string &varName,
                            const vartypePrintType_t printType = vartypePrintType_t::type) const;

      void printExtraDeclaration(printer &pout,
                                 const std::string &varName) const;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };

    io::output& operator << (io::output &out,
                             const vartype_t &type);

    printer& operator << (printer &pout,
                          const vartype_t &type);
  }
}

#endif
