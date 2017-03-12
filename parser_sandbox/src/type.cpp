#include "type.hpp"

namespace occa {
  namespace lang {
    //---[ Specifier ]------------------
    specifier::specifier(const int specType_) :
      name(),
      specType(specType_) {}

    specifier::specifier(const std::string &name_, const int specType_) :
      name(name_),
      specType(specType_) {}

    specifier::~specifier() {}

    void specifier::printOn(std::string &out) const {
      out += name;
    }

    std::ostream& operator << (std::ostream &out, const specifier &s) {
      out << s.toString();
      return out;
    }

    //---[ Qualifier ]------------------
    qualifier::qualifier(const std::string &name_) :
      specifier(name_, specifier::qualifierType) {}

    qualifier::qualifier(const std::string &name_, const int specType_) :
      specifier(name_, specType_) {}

    qualifier::~qualifier() {}

    //---[ Qualifiers ]-----------------
    qualifiers::qualifiers() {}

    qualifiers::qualifiers(const qualifier &q) {
      add(q);
    }

    qualifiers::~qualifiers() {}

    int qualifiers::has(const qualifier &q) {
      const int count = (int) qualifierVec.size();
      if (count) {
        const qualifier **qs = &(qualifierVec[0]);
        for (int i = 0; i < count; ++i) {
          if (qs[i] == &q) {
            return i;
          }
        }
      }
      return -1;
    }

    void qualifiers::add(const qualifier &q) {
      qualifierVec.push_back(&q);
    }

    void qualifiers::remove(const qualifier &q) {
      const int idx = has(q);
      if (idx >= 0) {
        qualifierVec.erase(qualifierVec.begin() + idx);
      }
    }

    std::string qualifiers::toString() const {
      std::string s;
      printOn(s);
      return s;
    }

    void qualifiers::printOn(std::string &out) const {
      const int count = (int) qualifierVec.size();
      if (!count) {
        return;
      }

      qualifierVec[0]->printOn(out);
      for (int i = 1; i < count; ++i) {
        out += ' ';
        qualifierVec[i]->printOn(out);
      }
    }

    std::ostream& operator << (std::ostream &out, const qualifiers &qs) {
      out << qs.toString();
      return out;
    }

    //---[ Type ]-----------------------
    type::type() :
      specifier(specifier::definedType),
      baseType(NULL) {}

    type::type(const std::string &name_) :
      specifier(name_, specifier::definedType),
      baseType(NULL) {}

    type::type(const std::string &name_,
               const int specType_) :
      specifier(name_, specType_),
      baseType(NULL) {}

    type::type(const type &baseType_) :
      specifier(specifier::definedType),
      baseType(&(baseType_.clone())) {}

    type::type(const qualifiers &qs) :
      specifier(specifier::definedType),
      baseType(NULL),
      qualifiers_(qs) {}

    type::type(const qualifiers &qs, const type &baseType_) :
      specifier(specifier::definedType),
      baseType(&(baseType_.clone())),
      qualifiers_(qs) {}

    type::~type() {
      if (baseType && baseType->isUnnamed()) {
        delete baseType;
      }
    }

    type& type::clone() const {
      if (isNamed()) {
        return *(const_cast<type*>(this));
      }
      if (baseType) {
        return *(new type(qualifiers_, baseType->clone()));
      }
      return *(new type(qualifiers_));
    }

    void type::printOn(std::string &out) const {
      if (qualifiers_.size()) {
        qualifiers_.printOn(out);
        if (baseType || isNamed()) {
          out += ' ';
        }
      }
      if (baseType) {
        baseType->printOn(out);
      } else {
        out += name;
      }
    }

    //---[ Primitive ]------------------
    primitive::primitive(const std::string &name_) :
      type(name_, specifier::primitiveType) {}

    primitive::~primitive() {}

    type& primitive::clone() const {
      return *(const_cast<primitive*>(this));
    }

    //---[ Pointer ]--------------------
    pointer::pointer(const type &t) :
      type(t) {}

    pointer::pointer(const qualifiers &qs) :
      type(qs) {}

    pointer::pointer(const qualifiers &qs, const type &t) :
      type(qs, t) {}

    pointer::~pointer() {}

    type& pointer::clone() const {
      if (baseType) {
        return *(new pointer(qualifiers_, baseType->clone()));
      }
      return *(new pointer(qualifiers_));
    }

    void pointer::printOn(std::string &out) const {
      if (!baseType) {
        return;
      }
      baseType->printOn(out);
      out += " *";
      if (qualifiers_.size()) {
        out += ' ';
        qualifiers_.printOn(out);
      }
    }

    //---[ Typedef ]--------------------
    typedefType::~typedefType() {}

    type& typedefType::clone() const {
      return *(const_cast<typedefType*>(this));
    }

    //---[ Class ]----------------------
    classType::~classType() {}

    type& classType::clone() const {
      return *(const_cast<classType*>(this));
    }
  }
}
