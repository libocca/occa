#include <sstream>

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"

#include "type.hpp"
#include "statement.hpp"
#include "variable.hpp"
#include "expression.hpp"

namespace occa {
  namespace lang {
    //---[ Specifier ]------------------
    specifier::specifier() :
      name() {}

    specifier::specifier(const std::string &name_) :
      name(name_) {}

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
      std::cout << toString();
    }

    //---[ Qualifier ]------------------
    qualifier::qualifier(const std::string &name_,
                         const qtype_t qtype_) :
      specifier(name_),
      qtype(qtype_) {}

    qualifier::~qualifier() {}

    stype_t qualifier::type() const {
      return specifierType::qualifier;
    }
    //==================================

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
      std::cout << toString();
    }
    //==================================

    //---[ Type ]-----------------------
    type_t::type_t() :
      baseType(NULL) {}

    type_t::type_t(const std::string &name_) :
      specifier(name_),
      baseType(NULL) {}

    type_t::type_t(const type_t &baseType_,
                   const std::string &name_) :
      specifier(name_),
      baseType(&baseType_) {}

    type_t::type_t(const qualifiers &qs,
                   const std::string &name_) :
      specifier(name_),
      baseType(NULL),
      qualifiers_(qs) {}

    type_t::type_t(const qualifiers &qs,
                   const type_t &baseType_,
                   const std::string &name_) :
      specifier(name_),
      baseType(&baseType_),
      qualifiers_(qs) {}

    type_t::~type_t() {
      if (baseType && baseType->isUnnamed()) {
        // TODO: Fix baseType deletion
        // delete baseType;
      }
    }

    stype_t type_t::type() const {
      return specifierType::type;
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

    void type_t::printLeft(printer_t &pout) const {
      if (qualifiers_.size()) {
        qualifiers_.print(pout);
        pout << ' ';
      }
      if (baseType) {
        baseType->printLeft(pout);
      }
      if (name.size()) {
        if (pout.lastCharNeedsWhitespace()) {
          pout << ' ';
        }
        pout << name;
      }
    }

    void type_t::printRight(printer_t &pout) const {
      if (baseType) {
        baseType->printRight(pout);
      }
    }

    void type_t::print(printer_t &pout) const {
      printLeft(pout);
      printRight(pout);
    }

    void type_t::print(printer_t &pout,
                       const variable_t &var) const {
      printLeft(pout);
      if (var.name.size()) {
        if (pout.lastCharNeedsWhitespace()) {
          pout << ' ';
        }
        pout << var.name;
      }
      printRight(pout);
    }

    std::string declarationType_t::declarationToString() const {
      std::stringstream ss;
      printer_t pout(ss);
      printDeclaration(pout);
      return ss.str();
    }

    void declarationType_t::declarationDebugPrint() const {
      std::cout << declarationToString();
    }
    //==================================

    //---[ Primitive ]------------------
    primitiveType::primitiveType(const std::string &name_) :
      type_t(name_) {}

    primitiveType::~primitiveType() {}

    stype_t primitiveType::type() const {
      return specifierType::primitive;
    }

    type_t& primitiveType::clone() const {
      return *(const_cast<primitiveType*>(this));
    }
    //==================================

    //---[ Pointer ]--------------------
    pointerType::pointerType(const type_t &baseType_) :
      type_t(baseType_) {}

    pointerType::pointerType(const pointerType &baseType_) :
      type_t(baseType_) {}

    pointerType::pointerType(const type_t &baseType_,
                             const qualifiers &rightQualifiers_) :
      type_t(baseType_),
      rightQualifiers(rightQualifiers_) {}

    pointerType::~pointerType() {}

    stype_t pointerType::type() const {
      return specifierType::pointer;
    }

    type_t& pointerType::clone() const {
      OCCA_ERROR("occa::lang::pointerType has a NULL baseType",
                 baseType);
      return *(new pointerType(baseType->clone(),
                               rightQualifiers));
    }

    void pointerType::printLeft(printer_t &pout) const {
      OCCA_ERROR("occa::lang::pointerType has a NULL baseType",
                 baseType);
      baseType->printLeft(pout);
      if (pout.lastCharNeedsWhitespace()) {
        pout << ' ';
      }
      pout << '*';
      if (rightQualifiers.size()) {
        pout << ' ';
        rightQualifiers.print(pout);
      }
    }
    //==================================

    //---[ Array ]----------------------
    arrayType::arrayType(const type_t &baseType_) :
      type_t(baseType_),
      size(new emptyNode()) {}

    arrayType::arrayType(const arrayType &baseType_) :
      type_t(baseType_),
      size(new emptyNode()) {}

    arrayType::arrayType(const type_t &baseType_,
                         const exprNode &size_) :
      type_t(baseType_),
      size(&(size_.clone())) {}

    arrayType::~arrayType() {
      delete size;
    }

    stype_t arrayType::type() const {
      return specifierType::array;
    }

    type_t& arrayType::clone() const {
      OCCA_ERROR("occa::lang::arrayType has a NULL baseType",
                 baseType);
      return *(new arrayType(baseType->clone(),
                             size->clone()));
    }

    void arrayType::setSize(exprNode &size_) {
      size = &(size_.clone());
    }

    void arrayType::printRight(printer_t &pout) const {
      baseType->printRight(pout);
      pout << '[';
      size->print(pout);
      pout << ']';
    }
    //==================================

    //---[ Reference ]------------------
    referenceType::referenceType(const type_t &baseType_) :
      type_t(baseType_) {}

    referenceType::referenceType(const referenceType &baseType_) :
      type_t(baseType_) {}

    referenceType::~referenceType() {}

    stype_t referenceType::type() const {
      return specifierType::reference;
    }

    type_t& referenceType::clone() const {
      OCCA_ERROR("occa::lang::referenceType has a NULL baseType",
                 baseType);
      return *(new referenceType(baseType->clone()));
    }

    void referenceType::printLeft(printer_t &pout) const {
      OCCA_ERROR("occa::lang::referenceType has a NULL baseType",
                 baseType);
      baseType->printLeft(pout);
      if (pout.lastCharNeedsWhitespace()) {
        pout << ' ';
      }
      pout << '&';
    }
    //==================================

    //---[ Class ]----------------------
    classType::classType(const std::string &name_,
                         const int label_) :
      type_t(name_),
      label(label_),
      body(NULL) {}

    classType::~classType() {
      if (body) {
        delete body;
      }
    }

    stype_t classType::type() const {
      return specifierType::class_;
    }

    type_t& classType::clone() const {
      return *(const_cast<classType*>(this));
    }

    void classType::printDeclaration(printer_t &pout) const {
      pout.printIndentation();

      switch (label) {
      case classLabel::class_ : pout << "class" ; break;
      case classLabel::enum_  : pout << "enum"  ; break;
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
    //==================================

    //---[ Typedef ]--------------------
    typedefType::typedefType(const type_t &baseType_,
                             const std::string &name_) :
      type_t(baseType_, name_) {}

    typedefType::typedefType(const qualifiers &qs,
                             const type_t &baseType_,
                             const std::string &name_) :
      type_t(qs, baseType_, name_) {}

    typedefType::~typedefType() {}

    stype_t typedefType::type() const {
      return specifierType::typedef_;
    }

    type_t& typedefType::clone() const {
      return *(const_cast<typedefType*>(this));
    }

    void typedefType::printLeft(printer_t &pout) const {
      pout << name;
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
    //==================================

    //---[ Function ]-------------------
    functionType::functionType(const type_t &returnType) :
      type_t(returnType) {}

    functionType::functionType(const functionType &returnType) :
      type_t(returnType) {}

    functionType::functionType(const type_t &returnType,
                               const std::string &name_) :
      type_t(returnType, name_) {}

    functionType::~functionType() {
      const int argCount = argumentCount();
      if (argCount) {
        type_t **args_ = &(args[0]);
        for (int i = 0; i < argCount; ++i) {
          if (args_[i]->isUnnamed()) {
            // TODO: Fix args deletion
            // delete args_[i];
          }
        }
      }
    }

    stype_t functionType::type() const {
      return specifierType::function;
    }

    const type_t& functionType::returnType() const {
      OCCA_ERROR("occa::lang::functionType has a NULL baseType",
                 baseType);
      return *(baseType);
    }

    void functionType::setReturnType(const type_t &returnType) {
      baseType = &returnType;
    }

    void functionType::addArgument(const type_t &argType,
                                   const std::string &argName) {
      args.push_back(new type_t(argType, argName));
    }

    void functionType::addArgument(const qualifiers &qs,
                                   const type_t &argType,
                                   const std::string &argName) {
      args.push_back(new type_t(qs, argType, argName));
    }

    void functionType::printDeclarationLeft(printer_t &pout) const {
      if (baseType->type() & specifierType::function) {
        dynamic_cast<const functionType*>(baseType)->
          printDeclarationLeft(pout);
      } else {
        baseType->print(pout);
      }
      if (pout.lastCharNeedsWhitespace() &&
          !(baseType->type() & specifierType::function)) {
        pout << ' ';
      }
      pout << "(*";
    }

    void functionType::printDeclarationRight(printer_t &pout) const {
      pout << ")(";
      const std::string argIndent = pout.indentFromNewline();
      const int argCount = argumentCount();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ",\n" << argIndent;
        }
        args[i]->print(pout);
      }
      pout << ')';
      if (baseType->type() & specifierType::function) {
        dynamic_cast<const functionType*>(baseType)->
          printDeclarationRight(pout);
      }
      pout << '\n';
    }

    void functionType::printDeclaration(printer_t &pout) const {
      OCCA_ERROR("occa::lang::functionType has a NULL baseType",
                 baseType);
      pout.printIndentation();
      printDeclarationLeft(pout);
      if (isNamed()) {
        pout << name;
      }
      printDeclarationRight(pout);
    }
    //==================================

    //---[ Attribute ]------------------
    attribute::attribute(const std::string &name_) :
      specifier(name_) {}

    attribute::~attribute() {}

    stype_t attribute::type() const {
      return specifierType::attribute;
    }

    void attribute::print(printer_t &pout) const {}
    //==================================
  }
}
