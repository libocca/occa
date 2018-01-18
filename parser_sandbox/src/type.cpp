/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <sstream>

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"

#include "type.hpp"
#include "typeBuiltins.hpp"
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

    void specifier::print(printer &pout) const {
      pout << name;
    }

    std::string specifier::toString() const {
      std::stringstream ss;
      printer pout(ss);
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
    qualifiers_t::qualifiers_t() {}

    qualifiers_t::qualifiers_t(const qualifier &q) {
      add(q);
    }

    qualifiers_t::~qualifiers_t() {}

    int qualifiers_t::indexOf(const qualifier &q) const {
      const int count = (int) qualifiers.size();
      if (count) {
        const qualifier * const *qs = &(qualifiers[0]);
        for (int i = 0; i < count; ++i) {
          if (qs[i] == &q) {
            return i;
          }
        }
      }
      return -1;
    }

    bool qualifiers_t::has(const qualifier &q) const {
      return (indexOf(q) >= 0);
    }

    void qualifiers_t::add(const qualifier &q) {
      qualifiers.push_back(&q);
    }

    void qualifiers_t::remove(const qualifier &q) {
      const int idx = indexOf(q);
      if (idx >= 0) {
        qualifiers.erase(qualifiers.begin() + idx);
      }
    }

    void qualifiers_t::print(printer &pout) const {
      const int count = (int) qualifiers.size();
      if (!count) {
        return;
      }
      qualifiers[0]->print(pout);
      for (int i = 1; i < count; ++i) {
        pout << ' ';
        qualifiers[i]->print(pout);
      }
    }

    std::string qualifiers_t::toString() const {
      std::stringstream ss;
      printer pout(ss);
      print(pout);
      return ss.str();
    }

    void qualifiers_t::debugPrint() const {
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

    type_t::type_t(const qualifiers_t &qualifiers_,
                   const std::string &name_) :
      specifier(name_),
      baseType(NULL),
      qualifiers(qualifiers_) {}

    type_t::type_t(const qualifiers_t &qualifiers_,
                   const type_t &baseType_,
                   const std::string &name_) :
      specifier(name_),
      baseType(&baseType_),
      qualifiers(qualifiers_) {}

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
        return *(new type_t(qualifiers, baseType->clone()));
      }
      return *(new type_t(qualifiers));
    }

    bool type_t::canBeDereferenced() const {
      if (baseType) {
        return baseType->canBeDereferenced();
      }
      return false;
    }

    bool type_t::canBeCastedToExplicitly(const type_t &alias) const {
      if (baseType) {
        return baseType->canBeCastedToExplicitly(alias);
      }
      return false;
    }

    bool type_t::canBeCastedToImplicitly(const type_t &alias) const {
      if (baseType) {
        return baseType->canBeCastedToImplicitly(alias);
      }
      return false;
    }

    void type_t::printLeft(printer &pout) const {
      if (qualifiers.size()) {
        qualifiers.print(pout);
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

    void type_t::printRight(printer &pout) const {
      if (baseType) {
        baseType->printRight(pout);
      }
    }

    void type_t::print(printer &pout) const {
      printLeft(pout);
      printRight(pout);
    }

    void type_t::print(printer &pout,
                       const variable &var) const {
      printLeft(pout);
      if (var.name.size()) {
        if (pout.lastCharNeedsWhitespace()) {
          pout << ' ';
        }
        pout << var.name;
      }
      printRight(pout);
    }

    std::string declarationType::declarationToString() const {
      std::stringstream ss;
      printer pout(ss);
      printDeclaration(pout);
      return ss.str();
    }

    void declarationType::declarationDebugPrint() const {
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

    bool primitiveType::canBeDereferenced() const {
      return false;
    }

    bool primitiveType::canBeCastedToExplicitly(const type_t &alias) const {
      const int aliasType = alias.type();
      if (aliasType == specifierType::reference) {
        return ((alias.baseType->type() == specifierType::primitive) &&
                (name == alias.baseType->name));
      }
      if (aliasType & (specifierType::typedef_ |
                       specifierType::reference)) {
        return canBeCastedToExplicitly(*alias.baseType);
      }
      return true;
    }

    bool primitiveType::canBeCastedToImplicitly(const type_t &alias) const {
      switch (alias.type()) {
      case specifierType::primitive:
        return true;
      case specifierType::pointer:
      case specifierType::array:
        return false;
      case specifierType::reference: {
        return ((alias.baseType->type() == specifierType::primitive) &&
                (name == alias.baseType->name));
      }
      case specifierType::type:
      case specifierType::typedef_:
        return canBeCastedToImplicitly(*alias.baseType);
      case specifierType::class_: // TODO: class type casting
      case specifierType::function:
        return false;
      }
      return false;
    }
    //==================================

    //---[ Base Pointer ]---------------
    bool basePointerType::canBeDereferenced() const {
      return true;
    }

    bool basePointerType::canBeCastedToExplicitly(const type_t &alias) const {
      // TODO
      //   - Class constructors
      //   - Primitives with the same size
      return alias.canBeDereferenced();
    }

    bool basePointerType::canBeCastedToImplicitly(const type_t &alias) const {
      // TODO
      //   - Class constructors
      //   - Primitives with the same size
      if (!alias.canBeDereferenced()) {
        return false;
      }
      const type_t &base      = *baseType;
      const type_t &aliasBase = *(alias.baseType);
      bool baseIsPtr      = base.canBeDereferenced();
      bool aliasBaseIsPtr = aliasBase.canBeDereferenced();
      if (baseIsPtr == aliasBaseIsPtr) {
        if (baseIsPtr) {
          return base.canBeCastedToImplicitly(aliasBase);
        }
        return (base.qualifiers.has(const_) == aliasBase.qualifiers.has(const_));
      }
      return false;
    }
    //==================================

    //---[ Pointer ]--------------------
    pointerType::pointerType(const type_t &baseType_) :
      type_t(baseType_) {}

    pointerType::pointerType(const qualifiers_t &qualifiers_,
                             const type_t &baseType_) :
      type_t(qualifiers_, baseType_) {}

    pointerType::pointerType(const pointerType &baseType_) :
      type_t(baseType_) {}

    pointerType::~pointerType() {}

    stype_t pointerType::type() const {
      return specifierType::pointer;
    }

    type_t& pointerType::clone() const {
      OCCA_ERROR("occa::lang::pointerType has a NULL baseType",
                 baseType);
      return *(new pointerType(qualifiers,
                               baseType->clone()));
    }

    void pointerType::printLeft(printer &pout) const {
      OCCA_ERROR("occa::lang::pointerType has a NULL baseType",
                 baseType);
      baseType->printLeft(pout);
      if (pout.lastCharNeedsWhitespace()) {
        pout << ' ';
      }
      pout << '*';
      if (qualifiers.size()) {
        pout << ' ';
        qualifiers.print(pout);
      }
    }
    //==================================

    //---[ Array ]----------------------
    arrayType::arrayType(const type_t &baseType_) :
      type_t(baseType_),
      size(new emptyNode()) {}

    arrayType::arrayType(const qualifiers_t &qualifiers_,
                         const type_t &baseType_) :
      type_t(qualifiers_, baseType_),
      size(new emptyNode()) {}

    arrayType::arrayType(const type_t &baseType_,
                         const exprNode &size_) :
      type_t(baseType_),
      size(&(size_.clone())) {}

    arrayType::arrayType(const qualifiers_t &qualifiers_,
                         const type_t &baseType_,
                         const exprNode &size_) :
      type_t(qualifiers_, baseType_),
      size(&(size_.clone())) {}

    arrayType::arrayType(const arrayType &baseType_) :
      type_t(baseType_),
      size(new emptyNode()) {}

    arrayType::~arrayType() {
      delete size;
    }

    stype_t arrayType::type() const {
      return specifierType::array;
    }

    type_t& arrayType::clone() const {
      OCCA_ERROR("occa::lang::arrayType has a NULL baseType",
                 baseType);
      return *(new arrayType(qualifiers,
                             baseType->clone(),
                             size->clone()));
    }

    void arrayType::setSize(exprNode &size_) {
      size = &(size_.clone());
    }

    void arrayType::printRight(printer &pout) const {
      baseType->printRight(pout);
      pout << '[';
      size->print(pout);
      pout << ']';
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

    type_t& functionType::clone() const {
      return *(new functionType(baseType->clone(),
                                name));
    }

    bool functionType::canBeCastedToImplicitly(const type_t &alias) const {
      // TODO: Handle function casting
      return false;
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

    void functionType::addArgument(const qualifiers_t &qualifiers_,
                                   const type_t &argType,
                                   const std::string &argName) {
      args.push_back(new type_t(qualifiers_, argType, argName));
    }

    void functionType::printDeclarationLeft(printer &pout) const {
      if (baseType->type() & specifierType::function) {
        baseType
          ->to<functionType>()
          .printDeclarationLeft(pout);
      } else {
        baseType->print(pout);
      }
      if (pout.lastCharNeedsWhitespace() &&
          !(baseType->type() & specifierType::function)) {
        pout << ' ';
      }
      pout << "(*";
    }

    void functionType::printDeclarationRight(printer &pout) const {
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
        baseType
          ->to<functionType>()
          .printDeclarationRight(pout);
      }
      pout << '\n';
    }

    void functionType::printDeclaration(printer &pout) const {
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

    //---[ Reference ]------------------
    referenceType::referenceType(const type_t &baseType_) :
      type_t(baseType_) {}

    referenceType::referenceType(const qualifiers_t &qualifiers_,
                                 const type_t &baseType_) :
      type_t(qualifiers_, baseType_) {}

    referenceType::referenceType(const referenceType &baseType_) :
      type_t(baseType_) {}

    referenceType::~referenceType() {}

    stype_t referenceType::type() const {
      return specifierType::reference;
    }

    type_t& referenceType::clone() const {
      OCCA_ERROR("occa::lang::referenceType has a NULL baseType",
                 baseType);
      return *(new referenceType(qualifiers,
                                 baseType->clone()));
    }

    bool referenceType::canBeDereferenced() const {
      return false;
    }

    bool referenceType::canBeCastedToExplicitly(const type_t &alias) const {
      return baseType->canBeCastedToExplicitly(alias);
    }

    bool referenceType::canBeCastedToImplicitly(const type_t &alias) const {
      return baseType->canBeCastedToImplicitly(alias);
    }

    void referenceType::printLeft(printer &pout) const {
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

    classType::classType(const qualifiers_t &qualifiers_,
                         const std::string &name_,
                         const int label_) :
      type_t(qualifiers_, name_),
      label(label_),
      body(NULL) {}

    classType::classType(const std::string &name_,
                         const int label_,
                         blockStatement &body_) :
      type_t(name_),
      label(label_),
      body(&(body_.clone().to<blockStatement>())) {}

    classType::classType(const qualifiers_t &qualifiers_,
                         const std::string &name_,
                         const int label_,
                         blockStatement &body_) :
      type_t(qualifiers_, name_),
      label(label_),
      body(&(body_.clone().to<blockStatement>())) {}

    classType::~classType() {}

    stype_t classType::type() const {
      return specifierType::class_;
    }

    type_t& classType::clone() const {
      if (body) {
        return *(new classType(qualifiers,
                               name,
                               label,
                               body->clone().to<blockStatement>()));
      }
      return *(new classType(qualifiers, name, label));
    }

    bool classType::canBeDereferenced() const {
      return false;
    }

    bool classType::canBeCastedToExplicitly(const type_t &alias) const {
      // TODO: Handle class casting
      return false;
    }

    bool classType::canBeCastedToImplicitly(const type_t &alias) const {
      // TODO: Handle class casting
      return false;
    }

    void classType::printDeclaration(printer &pout) const {
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
      if (body) {
        pout.pushInlined(true);
        body->print(pout);
        pout.pushInlined(false);
      } else {
        pout << " {}";
      }
      pout << ";\n";
    }
    //==================================

    //---[ Typedef ]--------------------
    typedefType::typedefType(const type_t &baseType_,
                             const std::string &name_) :
      type_t(baseType_, name_) {}

    typedefType::typedefType(const qualifiers_t &qualifiers_,
                             const type_t &baseType_,
                             const std::string &name_) :
      type_t(qualifiers_, baseType_, name_) {}

    typedefType::~typedefType() {}

    stype_t typedefType::type() const {
      return specifierType::typedef_;
    }

    type_t& typedefType::clone() const {
      return *(new typedefType(qualifiers,
                               baseType->clone(),
                               name));
    }

    bool typedefType::canBeDereferenced() const {
      return baseType->canBeDereferenced();
    }

    bool typedefType::canBeCastedToExplicitly(const type_t &alias) const {
      return baseType->canBeCastedToExplicitly(alias);
    }

    bool typedefType::canBeCastedToImplicitly(const type_t &alias) const {
      return baseType->canBeCastedToImplicitly(alias);
    }

    void typedefType::printLeft(printer &pout) const {
      pout << name;
    }

    void typedefType::printDeclaration(printer &pout) const {
      OCCA_ERROR("occa::lang::typedefType has a NULL baseType",
                 baseType);
      pout.printIndentation();
      pout << "typedef ";
      if (qualifiers.size()) {
        qualifiers.print(pout);
        pout << ' ';
      }
      baseType->print(pout);
      pout << ' ' << name << ";\n";
    }
    //==================================

    //---[ Attribute ]------------------
    attribute_t::attribute_t(const std::string &name_) :
      specifier(name_) {}

    attribute_t::~attribute_t() {}

    stype_t attribute_t::type() const {
      return specifierType::attribute;
    }

    void attribute_t::print(printer &pout) const {}
    //==================================
  }
}
