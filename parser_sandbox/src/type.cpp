#include <sstream>

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"

#include "type.hpp"
#include "statement.hpp"

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

    std::string specifier::uniqueName() const {
      return name;
    }

    void specifier::print(printer_t &pout) const {
      pout << name;
    }

    std::string specifier::toString() const {
      std::stringstream ss;
      printer_t pout(ss);
      print(pout);
      return ss.str();
    }

    void specifier::debugPrint() const {
      printer_t pout(std::cout);
      print(pout);
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

    void qualifiers::print(printer_t &pout) const {
      const int count = (int) qualifierVec.size();
      if (!count) {
        return;
      }
      qualifierVec[0]->print(pout);
      for (int i = 1; i < count; ++i) {
        pout << ' ';
        qualifierVec[i]->print(pout);
      }
    }

    std::string qualifiers::toString() const {
      std::stringstream ss;
      printer_t pout(ss);
      print(pout);
      return ss.str();
    }

    void qualifiers::debugPrint() const {
      printer_t pout(std::cout);
      print(pout);
    }

    //---[ Type ]-----------------------
    type_t::type_t() :
      specifier(specifier::definedType),
      baseType(NULL) {}

    type_t::type_t(const std::string &name_) :
      specifier(name_, specifier::definedType),
      baseType(NULL) {}

    type_t::type_t(const std::string &name_,
               const int specType_) :
      specifier(name_, specType_),
      baseType(NULL) {}

    type_t::type_t(const type_t &baseType_, const std::string &name_) :
      specifier(name_, specifier::definedType),
      baseType(&(baseType_.clone())) {}

    type_t::type_t(const qualifiers &qs, const std::string &name_) :
      specifier(name_, specifier::definedType),
      baseType(NULL),
      qualifiers_(qs) {}

    type_t::type_t(const qualifiers &qs, const type_t &baseType_, const std::string &name_) :
      specifier(name_, specifier::definedType),
      baseType(&(baseType_.clone())),
      qualifiers_(qs) {}

    type_t::~type_t() {
      if (baseType && baseType->isUnnamed()) {
        delete baseType;
      }
    }

    void type_t::replaceBaseType(const type_t &baseType_) {
      if (baseType && baseType->isUnnamed()) {
        delete baseType;
      }
      baseType = &(baseType_.clone());
    }

    type_t& type_t::clone() const {
      if (isNamed()) {
        return *(const_cast<type_t*>(this));
      }
      if (baseType) {
        return *(new type_t(qualifiers_, baseType->clone()));
      }
      return *(new type_t(qualifiers_));
    }

    void type_t::print(printer_t &pout) const {
      if (qualifiers_.size()) {
        qualifiers_.print(pout);
        if (baseType || isNamed()) {
          pout << ' ';
        }
      }
      if (baseType) {
        baseType->print(pout);
        if (isNamed()) {
          pout << ' ';
        }
      }
      pout << name;
    }

    //---[ Primitive ]------------------
    primitiveType::primitiveType(const std::string &name_) :
      type_t(name_, specifier::primitiveType) {}

    primitiveType::~primitiveType() {}

    type_t& primitiveType::clone() const {
      return *(const_cast<primitiveType*>(this));
    }

    void primitiveType::print(printer_t &pout) const {
      pout << name;
    }

    //---[ Pointer ]--------------------
    pointerType::pointerType(const type_t &t) :
      type_t(t) {}

    pointerType::pointerType(const qualifiers &qs, const type_t &t) :
      type_t(qs, t) {}

    pointerType::~pointerType() {}

    type_t& pointerType::clone() const {
      OCCA_ERROR("occa::lang::pointerType has a NULL baseType",
                 baseType);
      return *(new pointerType(qualifiers_, baseType->clone()));
    }

    void pointerType::print(printer_t &pout) const {
      OCCA_ERROR("occa::lang::pointerType has a NULL baseType",
                 baseType);
      baseType->print(pout);
      pout << " *";
      if (qualifiers_.size()) {
        pout << ' ';
        qualifiers_.print(pout);
      }
    }

    //---[ Reference ]--------------------
    referenceType::referenceType(const type_t &t) :
      type_t(t) {}

    referenceType::~referenceType() {}

    type_t& referenceType::clone() const {
      OCCA_ERROR("occa::lang::referenceType has a NULL baseType",
                 baseType);
      return *(new referenceType(baseType->clone()));
    }

    void referenceType::print(printer_t &pout) const {
      OCCA_ERROR("occa::lang::referenceType has a NULL baseType",
                 baseType);
      baseType->print(pout);
      pout << " &";
      if (qualifiers_.size()) {
        pout << ' ';
        qualifiers_.print(pout);
      }
    }

    //---[ Class ]----------------------
    classType::classType(const std::string &name_,
                         const int label_) :
      name(name_),
      label(label_),
      body(NULL) {}

    classType::~classType() {
      if (body) {
        delete body;
      }
    }

    type_t& classType::clone() const {
      return *(const_cast<classType*>(this));
    }

    void classType::printDeclaration(printer_t &pout) const {
      pout.printIndentation();

      switch (label) {
      case classLabel::class_ : pout << "class" ; break;
      case classLabel::struct_: pout << "struct"; break;
      case classLabel::union_ : pout << "union" ; break;
      }
      if (name.size()) {
        pout << ' ' << name;
      }
      pout << " {";

      if (body) {
        pout << '\n';

        pout.pushInlined(false);
        pout.addIndentation();
        body->print(pout);
        pout.removeIndentation();
        pout.popInlined();

        pout.printIndentation();
      }

      pout << '}';
      if (!pout.isInlined()) {
        pout << ";\n";
      }
    }

    void classType::print(printer_t &pout) const {
      pout << name;
    }

    //---[ Typedef ]--------------------
    typedefType::typedefType(const type_t &t, const std::string &name_) :
      type_t(t, name_) {}

    typedefType::typedefType(const qualifiers &qs, const type_t &t, const std::string &name_) :
      type_t(qs, t, name_) {}

    typedefType::~typedefType() {}

    type_t& typedefType::clone() const {
      return *(const_cast<typedefType*>(this));
    }

    void typedefType::printDeclaration(printer_t &pout) const {
      OCCA_ERROR("occa::lang::typedefType has a NULL baseType",
                 baseType);
      pout.printIndentation();
      pout << "typedef ";
      if (qualifiers_.size()) {
        qualifiers_.print(pout);
        pout << ' ';
      }
      baseType->print(pout);
      pout << ' ' << name << ";\n";
    }

    void typedefType::print(printer_t &pout) const {
      pout << name;
    }

    //---[ Function ]-------------------
    functionType::functionType(const type_t &returnType_) :
      type_t(returnType_),
      body(NULL) {}

    functionType::functionType(const type_t &returnType_, const std::string &name_) :
      type_t(returnType_, name_),
      body(NULL) {}

    functionType::~functionType() {
      const int argCount = argumentCount();
      if (argCount) {
        type_t **args_        = &(args[0]);
        // void **defaultValues_ = &(defaultValues[0]);
        for (int i = 0; i < argCount; ++i) {
          if (args_[i]->isUnnamed()) {
            delete args_[i];
          }
          // if (defaultValues_[i]) {
          //   delete defaultValues[i];
          // }
        }
      }
      if (body) {
        delete body;
      }
    }

    void functionType::setReturnType(const type_t &returnType) {
      replaceBaseType(returnType);
    }

    const type_t& functionType::returnType() const {
      OCCA_ERROR("occa::lang::functionType has a NULL baseType",
                 baseType);
      return *(baseType);
    }

    void functionType::addArg(const type_t &argType,
                              const std::string &argName,
                              const void *defaultValue) {
      args.push_back(new type_t(argType, argName));
      // defaultValues.push_back(defaultValue);
    }

    void functionType::addArg(const qualifiers &qs,
                              const type_t &argType,
                              const std::string &argName,
                              const void *defaultValue) {
      args.push_back(new type_t(qs, argType, argName));
      // defaultValues.push_back(defaultValue);
    }

    void functionType::setBody(statement_t &body_) {
      body = &body_;
    }

    void functionType::printDeclaration(printer_t &pout) const {
      statement_t *originalBody = body;
      body = NULL;
      print(pout);
      body = originalBody;
    }

    void functionType::print(printer_t &pout) const {
      OCCA_ERROR("occa::lang::functionType has a NULL baseType",
                 baseType);
      pout.printIndentation();
      baseType->print(pout);
      pout << ' ' << name << " (";
      const std::string argIndent = pout.indentFromNewline();
      const int argCount = argumentCount();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ",\n" << argIndent;
        }
        args[i]->print(pout);

        // void *defaultValue = defaultValues[i];
        // if (body && defaultValue) {
        //   pout << " = ";
          // defaultValue->print(pout);
        // }
      }
      pout << ')';

      if (!body) {
        pout << ";\n";
      } else {
        pout << " {\n";

        pout.addIndentation();
        body->print(pout);
        pout.removeIndentation();

        pout.printIndentation();
        pout << "}\n";
      }
    }

    //---[ Attribute ]------------------
    attribute::attribute(const std::string &name_) :
      specifier(name_, specifier::attributeType) {}

    attribute::~attribute() {}

    void attribute::print(printer_t &pout) const {}
  }
}
