#include <occa/lang/builtins/types.hpp>
#include <occa/lang/expr/exprNode.hpp>
#include <occa/lang/token.hpp>
#include <occa/lang/type/array.hpp>
#include <occa/lang/type/pointer.hpp>
#include <occa/lang/type/vartype.hpp>

namespace occa {
  namespace lang {
    vartype_t::vartype_t() :
      typeToken(NULL),
      type(NULL),
      referenceToken(NULL),
      bitfield(-1) {}

    vartype_t::vartype_t(const type_t &type_) :
      typeToken(NULL),
      referenceToken(NULL),
      bitfield(-1) {

      if (!type_.isNamed()) {
        type = &(type_.clone());
      } else {
        type = &type_;
      }
    }

    vartype_t::vartype_t(const identifierToken &typeToken_,
                         const type_t &type_) :
      typeToken((identifierToken*) typeToken_.clone()),
      referenceToken(NULL),
      bitfield(-1) {

      if (!type_.isNamed()) {
        type = &(type_.clone());
      } else {
        type = &type_;
      }
    }

    vartype_t::vartype_t(const vartype_t &other) :
      typeToken(NULL),
      type(NULL),
      referenceToken(NULL),
      bitfield(-1) {
      *this = other;
    }

    vartype_t::~vartype_t() {
      clear();
    }

    vartype_t& vartype_t::operator = (const vartype_t &other) {
      clear();
      qualifiers = other.qualifiers;
      pointers   = other.pointers;
      arrays     = other.arrays;

      typeToken = (identifierToken*) token_t::clone(other.typeToken);

      if (other.type &&
          !other.type->isNamed()) {
        type = &(other.type->clone());
      } else {
        type = other.type;
      }

      if (other.referenceToken) {
        referenceToken = other.referenceToken->clone();
      } else {
        referenceToken = NULL;
      }

      bitfield = other.bitfield;

      customPrefix = other.customPrefix;
      customSuffix = other.customSuffix;

      return *this;
    }

    void vartype_t::clear() {
      qualifiers.clear();
      pointers.clear();
      arrays.clear();
      bitfield = -1;
      customPrefix = "";
      customSuffix = "";

      delete typeToken;
      typeToken = NULL;

      if (type &&
          !type->isNamed()) {
        delete type;
      }
      type = NULL;

      delete referenceToken;
      referenceToken = NULL;
    }

    bool vartype_t::isValid() const {
      return type;
    }

    bool vartype_t::isNamed() const {
      return typeToken;
    }

    std::string vartype_t::name() const {
      if (typeToken) {
        return typeToken->value;
      }
      return "";
    }

    fileOrigin vartype_t::origin() const {
      if (qualifiers.size()) {
        return qualifiers.qualifiers[0].origin;
      } else if (typeToken) {
        return typeToken->origin;
      } else if (type) {
        return type->source->origin;
      }
      return fileOrigin();
    }

    bool vartype_t::isPointerType() const {
      if (pointers.size()
          || arrays.size()) {
        return true;
      }
      return (type
              && type->isPointerType());
    }

    void vartype_t::setReferenceToken(token_t *token) {
      if (referenceToken
          && (referenceToken != token)) {
        delete referenceToken;
      }
      referenceToken = token->clone();
    }

    bool vartype_t::isReference() const {
      return referenceToken;
    }

    dtype_t vartype_t::dtype() const {
      if (!type) {
        return dtype::none;
      }

      dtype_t dtype = type->dtype();
      const int arrayCount = (int) arrays.size();
      for (int i = 0; i < arrayCount; ++i) {
        primitive primSize = arrays[i].size->evaluate();
        const int size = primSize.isNaN() ? -1 : primSize.to<int>();
        dtype = dtype_t::tuple(dtype, size);
      }

      return dtype;
    }

    bool vartype_t::operator == (const vartype_t &other) const {
      if (!type || !other.type) {
        return false;
      }

      vartype_t flat      = flatten();
      vartype_t otherFlat = other.flatten();

      if (((*flat.type)    != (*otherFlat.type)) ||
          (flat.qualifiers != otherFlat.qualifiers)) {
        return false;
      }

      const int pointerCount      = (int) flat.pointers.size();
      const int otherPointerCount = (int) otherFlat.pointers.size();

      const int arrayCount      = (int) flat.arrays.size();
      const int otherArrayCount = (int) otherFlat.arrays.size();

      if ((pointerCount + arrayCount)
          != (arrayCount + otherArrayCount)) {
        return false;
      }

      // Make sure qualifiers in pointers match
      // If there are extra pointers in one, make sure
      //   it doesn't have any qualifiers (e.g. int* == int[])
      vartype_t &maxFlat = ((pointerCount < otherPointerCount)
                            ? otherFlat
                            : flat);
      const int minPointerCount = ((pointerCount < otherPointerCount)
                                   ? pointerCount
                                   : otherPointerCount);
      const int maxPointerCount = ((pointerCount < otherPointerCount)
                                   ? otherPointerCount
                                   : pointerCount);

      for (int i = 0; i < minPointerCount; ++i) {
        if (flat.pointers[i].qualifiers
            != otherFlat.pointers[i].qualifiers) {
          return false;
        }
      }
      for (int i = minPointerCount; i < maxPointerCount; ++i) {
        if (maxFlat.pointers[i].qualifiers.size()) {
          return false;
        }
      }
      return true;
    }

    bool vartype_t::operator != (const vartype_t &other) const {
      return !(*this == other);
    }

    bool vartype_t::has(const qualifier_t &qualifier) const {
      return qualifiers.has(qualifier);
    }

    vartype_t& vartype_t::operator += (const qualifier_t &qualifier) {
      qualifiers += qualifier;
      return *this;
    }

    vartype_t& vartype_t::operator -= (const qualifier_t &qualifier) {
      qualifiers -= qualifier;
      return *this;
    }

    vartype_t& vartype_t::operator += (const qualifiers_t &qualifiers_) {
      qualifiers += qualifiers_;
      return *this;
    }

    void vartype_t::add(const fileOrigin &origin,
                        const qualifier_t &qualifier) {
      qualifiers.add(origin, qualifier);
    }

    void vartype_t::add(const qualifierWithSource &qualifier) {
      qualifiers.add(qualifier);
    }

    void vartype_t::add(const int index,
                        const fileOrigin &origin,
                        const qualifier_t &qualifier) {
      qualifiers.add(index, origin, qualifier);
    }

    void vartype_t::add(const int index,
                        const qualifierWithSource &qualifier) {
      qualifiers.add(index, qualifier);
    }

    vartype_t& vartype_t::operator += (const pointer_t &pointer) {
      pointers.push_back(pointer);
      return *this;
    }

    vartype_t& vartype_t::operator += (const pointerVector &pointers_) {
      const int pointerCount = (int) pointers_.size();
      for (int i = 0; i < pointerCount; ++i) {
        pointers.push_back(pointers_[i]);
      }
      return *this;
    }

    vartype_t& vartype_t::operator += (const array_t &array) {
      arrays.push_back(array);
      return *this;
    }

    vartype_t& vartype_t::operator += (const arrayVector &arrays_) {
      const int arrayCount = (int) arrays_.size();
      for (int i = 0; i < arrayCount; ++i) {
        arrays.push_back(arrays_[i]);
      }
      return *this;
    }

    bool vartype_t::hasAttribute(const std::string &attr) const {
      return (type
              ? type->hasAttribute(attr)
              : false);
    }

    vartype_t vartype_t::declarationType() const {
      vartype_t other;
      other.type = type;
      other.qualifiers = qualifiers;
      return other;
    }

    vartype_t vartype_t::flatten() const {
      if (!type ||
          (type->type() != typeType::typedef_)) {
        return *this;
      }

      vartype_t flat = (type
                        ->to<typedef_t>()
                        .baseType
                        .flatten());

      flat += qualifiers;
      flat += pointers;
      flat += arrays;

      // Remove typedef if it exists
      flat -= typedef_;

      return flat;
    }

    void vartype_t::printDeclaration(printer &pout,
                                     const std::string &varName,
                                     const bool printType) const {
      if (!type) {
        return;
      }

      if (customPrefix.size()) {
        pout << customPrefix << ' ';
      }

      bool spaceAfterName = false;
      if (printType) {
        if (qualifiers.size()) {
          pout << qualifiers << ' ';
        }
        pout << *type;
      }

      const int pointerCount = (int) pointers.size();
      if (printType && pointerCount) {
        spaceAfterName = true;
        pout << ' ';
      }

      for (int i = 0; i < pointerCount; ++i) {
        pout << pointers[i];
        // Don't add a space after the last * if possible
        if (pointers[i].qualifiers.size()) {
          pout << ' ';
        }
      }

      if (referenceToken) {
        if (printType && !spaceAfterName) {
          spaceAfterName = true;
          pout << ' ';
        }
        pout << '&';
      }

      if (varName.size()) {
        if (printType && !spaceAfterName) {
          spaceAfterName = true;
          pout << ' ';
        }
        pout << varName;
      }

      const int arrayCount = (int) arrays.size();
      for (int i = 0; i < arrayCount; ++i) {
        pout << arrays[i];
      }

      if (customSuffix.size()) {
        pout << ' ' << customSuffix;
      }

      if (bitfield >= 0) {
        pout << " : " << bitfield;
      }
    }

    void vartype_t::printExtraDeclaration(printer &pout,
                                          const std::string &varName) const {
      printDeclaration(pout, varName, false);
    }

    void vartype_t::printWarning(const std::string &message) const {
      fileOrigin origin_ = origin();
      if (origin_.isValid()) {
        origin_.printWarning(message);
      }
    }

    void vartype_t::printError(const std::string &message) const {
      fileOrigin origin_ = origin();
      if (origin_.isValid()) {
        origin_.printError(message);
      }
    }

    io::output& operator << (io::output &out,
                               const vartype_t &type) {
      printer pout(out);
      pout << type;
      return out;
    }

    printer& operator << (printer &pout,
                          const vartype_t &type) {
      type.printDeclaration(pout, "", true);
      return pout;
    }
  }
}
