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
    //---[ Types ]----------------------
    namespace specifierType {
      const ktype_t none = 0;

      const ktype_t qualifier = (1 << 0);
      const ktype_t type      = (1 << 1);
      const ktype_t primitive = (1 << 2);
      const ktype_t pointer   = (1 << 3);
      const ktype_t reference = (1 << 4);
      const ktype_t array     = (1 << 5);
      const ktype_t struct_   = (1 << 6);
      const ktype_t class_    = (1 << 7);
      const ktype_t typedef_  = (1 << 8);
      const ktype_t function  = (1 << 9);
      const ktype_t attribute = (1 << 10);
    }

    namespace qualifierType {
      const ktype_t none          = 0;

      const ktype_t auto_         = (1L << 0);
      const ktype_t const_        = (1L << 1);
      const ktype_t constexpr_    = (1L << 2);
      const ktype_t restrict_     = (1L << 3);
      const ktype_t signed_       = (1L << 4);
      const ktype_t unsigned_     = (1L << 5);
      const ktype_t volatile_     = (1L << 6);
      const ktype_t register_     = (1L << 7);
      const ktype_t typeInfo      = (const_     |
                                     constexpr_ |
                                     signed_    |
                                     unsigned_  |
                                     volatile_  |
                                     register_);

      const ktype_t extern_       = (1L << 8);
      const ktype_t static_       = (1L << 9);
      const ktype_t thread_local_ = (1L << 10);
      const ktype_t globalScope   = (extern_ |
                                     static_ |
                                     thread_local_);

      const ktype_t friend_       = (1L << 11);
      const ktype_t mutable_      = (1L << 12);
      const ktype_t classInfo     = (friend_ |
                                     mutable_);

      const ktype_t inline_       = (1L << 13);
      const ktype_t virtual_      = (1L << 14);
      const ktype_t explicit_     = (1L << 15);
      const ktype_t functionInfo  = (typeInfo |
                                     inline_  |
                                     virtual_ |
                                     explicit_);

      const ktype_t builtin_      = (1L << 16);
      const ktype_t typedef_      = (1L << 17);
      const ktype_t class_        = (1L << 18);
      const ktype_t enum_         = (1L << 19);
      const ktype_t struct_       = (1L << 20);
      const ktype_t union_        = (1L << 21);
      const ktype_t newType       = (typedef_ |
                                     class_   |
                                     enum_    |
                                     struct_  |
                                     union_);
    }

    namespace classAccess {
      const int private_   = (1 << 0);
      const int protected_ = (1 << 1);
      const int public_    = (1 << 2);
    }
    //==================================

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
                         const ktype_t ktype_) :
      specifier(name_),
      ktype(ktype_) {}

    qualifier::~qualifier() {}

    ktype_t qualifier::type() const {
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

    ktype_t type_t::type() const {
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

    ktype_t primitiveType::type() const {
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

    ktype_t pointerType::type() const {
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

    ktype_t arrayType::type() const {
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

    ktype_t functionType::type() const {
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

    ktype_t referenceType::type() const {
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

    //---[ Structure ]------------------
    const int structureType::class_  = (1 << 0);
    const int structureType::enum_   = (1 << 1);
    const int structureType::struct_ = (1 << 2);
    const int structureType::union_  = (1 << 3);

    structureType::structureType(const std::string &name_,
                                 const int stype_) :
      type_t(name_),
      stype(stype_),
      body(NULL) {}

    structureType::structureType(const qualifiers_t &qualifiers_,
                                 const std::string &name_,
                                 const int stype_) :
      type_t(qualifiers_, name_),
      stype(stype_),
      body(NULL) {}

    structureType::structureType(const std::string &name_,
                                 const int stype_,
                                 blockStatement &body_) :
      type_t(name_),
      stype(stype_),
      body(&(body_.clone().to<blockStatement>())) {}

    structureType::structureType(const qualifiers_t &qualifiers_,
                                 const std::string &name_,
                                 const int stype_,
                                 blockStatement &body_) :
      type_t(qualifiers_, name_),
      stype(stype_),
      body(&(body_.clone().to<blockStatement>())) {}

    structureType::~structureType() {}

    ktype_t structureType::type() const {
      return specifierType::class_;
    }

    type_t& structureType::clone() const {
      if (body) {
        return *(new structureType(qualifiers,
                                   name,
                                   stype,
                                   body->clone().to<blockStatement>()));
      }
      return *(new structureType(qualifiers, name, stype));
    }

    bool structureType::canBeDereferenced() const {
      return false;
    }

    bool structureType::canBeCastedToExplicitly(const type_t &alias) const {
      // TODO: Handle class casting
      return false;
    }

    bool structureType::canBeCastedToImplicitly(const type_t &alias) const {
      // TODO: Handle class casting
      return false;
    }

    void structureType::printDeclaration(printer &pout) const {
      pout.printIndentation();

      switch (stype) {
      case structureType::class_ : pout << "class" ; break;
      case structureType::enum_  : pout << "enum"  ; break;
      case structureType::struct_: pout << "struct"; break;
      case structureType::union_ : pout << "union" ; break;
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

    ktype_t typedefType::type() const {
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

    ktype_t attribute_t::type() const {
      return specifierType::attribute;
    }

    void attribute_t::print(printer &pout) const {
      pout << '@' << name;
    }
    //==================================

    void getSpecifiers(specifierTrie &specifiers) {
      specifiers.add(const_.name       , &const_);
      specifiers.add(constexpr_.name   , &constexpr_);
      specifiers.add(friend_.name      , &friend_);
      specifiers.add(typedef_.name     , &typedef_);
      specifiers.add(signed_.name      , &signed_);
      specifiers.add(unsigned_.name    , &unsigned_);
      specifiers.add(volatile_.name    , &volatile_);

      specifiers.add(extern_.name      , &extern_);
      specifiers.add(mutable_.name     , &mutable_);
      specifiers.add(register_.name    , &register_);
      specifiers.add(static_.name      , &static_);
      specifiers.add(thread_local_.name, &thread_local_);

      specifiers.add(explicit_.name    , &explicit_);
      specifiers.add(inline_.name      , &inline_);
      specifiers.add(virtual_.name     , &virtual_);

      specifiers.add(class_.name       , &class_);
      specifiers.add(enum_.name        , &enum_);
      specifiers.add(struct_.name      , &struct_);
      specifiers.add(union_.name       , &union_);

      specifiers.add(bool_.name        , &bool_);
      specifiers.add(char_.name        , &char_);
      specifiers.add(char16_t_.name    , &char16_t_);
      specifiers.add(char32_t_.name    , &char32_t_);
      specifiers.add(wchar_t_.name     , &wchar_t_);
      specifiers.add(short_.name       , &short_);
      specifiers.add(int_.name         , &int_);
      specifiers.add(long_.name        , &long_);
      specifiers.add(float_.name       , &float_);
      specifiers.add(double_.name      , &double_);
      specifiers.add(void_.name        , &void_);
      specifiers.add(auto_.name        , &auto_);
    }
  }
}
