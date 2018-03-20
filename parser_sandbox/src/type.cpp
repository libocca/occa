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
      const int none = 0;

      const int qualifier = (1 << 0);
      const int type      = (1 << 1);
      const int primitive = (1 << 2);
      const int pointer   = (1 << 3);
      const int reference = (1 << 4);
      const int array     = (1 << 5);
      const int struct_   = (1 << 6);
      const int class_    = (1 << 7);
      const int typedef_  = (1 << 8);
      const int function  = (1 << 9);
      const int attribute = (1 << 10);
    }

    namespace qualifierType {
      const int none          = 0;

      const int auto_         = (1L << 0);
      const int const_        = (1L << 1);
      const int constexpr_    = (1L << 2);
      const int restrict_     = (1L << 3);
      const int signed_       = (1L << 4);
      const int unsigned_     = (1L << 5);
      const int volatile_     = (1L << 6);
      const int long_         = (1L << 7);
      const int longlong_     = (1L << 8);
      const int register_     = (1L << 9);
      const int typeInfo      = (const_     |
                                 constexpr_ |
                                 signed_    |
                                 unsigned_  |
                                 volatile_  |
                                 long_      |
                                 longlong_  |
                                 register_);

      const int extern_       = (1L << 10);
      const int static_       = (1L << 11);
      const int thread_local_ = (1L << 12);
      const int globalScope   = (extern_ |
                                 static_ |
                                 thread_local_);

      const int friend_       = (1L << 13);
      const int mutable_      = (1L << 14);
      const int classInfo     = (friend_ |
                                 mutable_);

      const int inline_       = (1L << 15);
      const int virtual_      = (1L << 16);
      const int explicit_     = (1L << 17);
      const int functionInfo  = (typeInfo |
                                 inline_  |
                                 virtual_ |
                                 explicit_);

      const int builtin_      = (1L << 18);
      const int typedef_      = (1L << 19);
      const int class_        = (1L << 20);
      const int enum_         = (1L << 21);
      const int struct_       = (1L << 22);
      const int union_        = (1L << 23);
      const int newType       = (typedef_ |
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
    qualifier_t::qualifier_t(const std::string &name_,
                             const int ktype_) :
      specifier(name_),
      ktype(ktype_) {}

    qualifier_t::~qualifier_t() {}

    int qualifier_t::type() const {
      return specifierType::qualifier;
    }
    //==================================

    //---[ Qualifiers ]-----------------
    qualifiers_t::qualifiers_t() {}

    qualifiers_t::qualifiers_t(const qualifier_t &qualifier) {
      *this += qualifier;
    }

    qualifiers_t::~qualifiers_t() {}

    void qualifiers_t::clear() {
      qualifiers.clear();
    }

    const qualifier_t* qualifiers_t::operator [] (const int index) {
      if ((index < 0) ||
          (index >= (int) qualifiers.size())) {
        return NULL;
      }
      return qualifiers[index];
    }

    int qualifiers_t::indexOf(const qualifier_t &qualifier) const {
      const int count = (int) qualifiers.size();
      if (count) {
        const qualifier_t * const *qs = &(qualifiers[0]);
        for (int i = 0; i < count; ++i) {
          if (qs[i] == &qualifier) {
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
        if (!other.has(*qualifiers[i])) {
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
        qualifiers.push_back(&qualifier);
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
        *this += *(other.qualifiers[i]);
      }
      return *this;
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
      baseType(NULL) {

      setBaseType(baseType_);
    }

    type_t::type_t(const qualifiers_t &qualifiers_,
                   const std::string &name_) :
      specifier(name_),
      baseType(NULL),
      qualifiers(qualifiers_) {}

    type_t::type_t(const qualifiers_t &qualifiers_,
                   const type_t &baseType_,
                   const std::string &name_) :
      specifier(name_),
      baseType(NULL),
      qualifiers(qualifiers_) {

      setBaseType(baseType_);
    }

    type_t::~type_t() {
      shallowFree(baseType);
    }

    void type_t::setBaseType(const type_t &baseType_) {
      if (baseType == &baseType_) {
        return;
      }

      shallowFree(baseType);

      if (baseType_.isNamed()) {
        baseType = &baseType_;
      } else {
        baseType = &(baseType_.shallowClone());
      }
    }

    int type_t::type() const {
      return specifierType::type;
    }

    type_t& type_t::clone() const {
      return *(new type_t(qualifiers, *baseType, name));
    }

    type_t& type_t::shallowClone() const {
      if (isNamed()) {
        return *(const_cast<type_t*>(this));
      }
      return *(new type_t(qualifiers, *baseType));
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

    declarationType::~declarationType() {}

    std::string declarationType::declarationToString() const {
      std::stringstream ss;
      printer pout(ss);
      printDeclaration(pout);
      return ss.str();
    }

    void declarationType::declarationDebugPrint() const {
      std::cout << declarationToString();
    }

    void shallowFree(const type_t *&type) {
      if (type) {
        if (type->isUnnamed()) {
          delete type;
        }
        type = NULL;
      }
    }
    //==================================

    //---[ Primitive ]------------------
    primitive_t::primitive_t(const std::string &name_) :
      type_t(name_) {}

    primitive_t::~primitive_t() {}

    int primitive_t::type() const {
      return specifierType::primitive;
    }

    type_t& primitive_t::clone() const {
      return *(const_cast<primitive_t*>(this));
    }

    bool primitive_t::canBeDereferenced() const {
      return false;
    }

    bool primitive_t::canBeCastedToExplicitly(const type_t &alias) const {
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

    bool primitive_t::canBeCastedToImplicitly(const type_t &alias) const {
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
    basePointer_t::~basePointer_t() {}

    bool basePointer_t::canBeDereferenced() const {
      return true;
    }

    bool basePointer_t::canBeCastedToExplicitly(const type_t &alias) const {
      // TODO
      //   - Class constructors
      //   - Primitives with the same size
      return alias.canBeDereferenced();
    }

    bool basePointer_t::canBeCastedToImplicitly(const type_t &alias) const {
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
    pointer_t::pointer_t(const type_t &baseType_) :
      type_t(baseType_) {}

    pointer_t::pointer_t(const qualifiers_t &qualifiers_,
                         const type_t &baseType_) :
      type_t(qualifiers_, baseType_) {}

    pointer_t::pointer_t(const pointer_t &baseType_) :
      type_t(baseType_) {}

    pointer_t::~pointer_t() {}

    int pointer_t::type() const {
      return specifierType::pointer;
    }

    type_t& pointer_t::clone() const {
      OCCA_ERROR("occa::lang::pointer_t has a NULL baseType",
                 baseType);
      return *(new pointer_t(qualifiers, *baseType));
    }

    void pointer_t::printLeft(printer &pout) const {
      OCCA_ERROR("occa::lang::pointer_t has a NULL baseType",
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
    array_t::array_t(const type_t &baseType_) :
      type_t(baseType_),
      size(new emptyNode()) {}

    array_t::array_t(const qualifiers_t &qualifiers_,
                     const type_t &baseType_) :
      type_t(qualifiers_, baseType_),
      size(new emptyNode()) {}

    array_t::array_t(const type_t &baseType_,
                     const exprNode &size_) :
      type_t(baseType_),
      size(&(size_.clone())) {}

    array_t::array_t(const qualifiers_t &qualifiers_,
                     const type_t &baseType_,
                     const exprNode &size_) :
      type_t(qualifiers_, baseType_),
      size(&(size_.clone())) {}

    array_t::array_t(const array_t &baseType_) :
      type_t(baseType_),
      size(new emptyNode()) {}

    array_t::~array_t() {
      delete size;
    }

    int array_t::type() const {
      return specifierType::array;
    }

    type_t& array_t::clone() const {
      OCCA_ERROR("occa::lang::array_t has a NULL baseType",
                 baseType);
      return *(new array_t(qualifiers,
                           *baseType,
                           size->clone()));
    }

    void array_t::setSize(exprNode &size_) {
      size = &(size_.clone());
    }

    void array_t::printRight(printer &pout) const {
      baseType->printRight(pout);
      pout << '[';
      size->print(pout);
      pout << ']';
    }
    //==================================

    //---[ Function ]-------------------
    function_t::function_t(const type_t &returnType) :
      type_t(returnType),
      isBlock(false) {
    }

    function_t::function_t(const type_t &returnType,
                           const std::string &name_) :
      type_t(returnType, name_),
      isBlock(false) {
    }

    function_t::function_t(const type_t &returnType,
                           const std::string &name_,
                           const argVector_t &args_) :
      type_t(returnType, name_),
      isBlock(false) {

      const int argCount = (int) args_.size();
      for (int i = 0; i < argCount; ++i) {
        args.push_back(&(args_[i]->clone()));
      }
    }

    function_t::~function_t() {
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        delete args[i];
      }
    }

    int function_t::type() const {
      return specifierType::function;
    }

    type_t& function_t::clone() const {
      return *(new function_t(*baseType, name, args));
    }

    bool function_t::canBeCastedToImplicitly(const type_t &alias) const {
      // TODO: Handle function casting
      return false;
    }

    const type_t& function_t::returnType() const {
      OCCA_ERROR("occa::lang::function_t has a NULL baseType",
                 baseType);
      return *(baseType);
    }

    void function_t::setReturnType(const type_t &returnType) {
      setBaseType(returnType);
    }

    void function_t::addArgument(const type_t &argType,
                                 const std::string &argName) {
      args.push_back(new variable(argType, argName));
    }

    void function_t::addArgument(const qualifiers_t &qualifiers_,
                                 const type_t &argType,
                                 const std::string &argName) {

      args.push_back(new variable(type_t(qualifiers_, argType),
                                  argName));
    }

    void function_t::printDeclarationLeft(printer &pout) const {
      if (baseType->type() & specifierType::function) {
        baseType
          ->to<function_t>()
          .printDeclarationLeft(pout);
      } else {
        baseType->print(pout);
      }
      if (pout.lastCharNeedsWhitespace() &&
          !(baseType->type() & specifierType::function)) {
        pout << ' ';
      }
      if (!isBlock) {
        pout << "(*";
      } else {
        pout << "(^";
      }
    }

    void function_t::printDeclarationRight(printer &pout) const {
      pout << ")(";
      const std::string argIndent = pout.indentFromNewline();
      const int argCount = argumentCount();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ",\n" << argIndent;
        }
        args[i]->printDeclaration(pout);
      }
      pout << ')';
      if (baseType->type() & specifierType::function) {
        baseType
          ->to<function_t>()
          .printDeclarationRight(pout);
      }
      pout << '\n';
    }

    void function_t::printDeclaration(printer &pout) const {
      OCCA_ERROR("occa::lang::function_t has a NULL baseType",
                 baseType);
      pout.printIndentation();
      printDeclarationLeft(pout);
      pout << name;
      printDeclarationRight(pout);
    }
    //==================================

    //---[ Reference ]------------------
    reference_t::reference_t(const type_t &baseType_) :
      type_t(baseType_) {}

    reference_t::reference_t(const qualifiers_t &qualifiers_,
                             const type_t &baseType_) :
      type_t(qualifiers_, baseType_) {}

    reference_t::reference_t(const reference_t &baseType_) :
      type_t(baseType_) {}

    reference_t::~reference_t() {}

    int reference_t::type() const {
      return specifierType::reference;
    }

    type_t& reference_t::clone() const {
      OCCA_ERROR("occa::lang::reference_t has a NULL baseType",
                 baseType);
      return *(new reference_t(qualifiers, *baseType));
    }

    bool reference_t::canBeDereferenced() const {
      return false;
    }

    bool reference_t::canBeCastedToExplicitly(const type_t &alias) const {
      return baseType->canBeCastedToExplicitly(alias);
    }

    bool reference_t::canBeCastedToImplicitly(const type_t &alias) const {
      return baseType->canBeCastedToImplicitly(alias);
    }

    void reference_t::printLeft(printer &pout) const {
      OCCA_ERROR("occa::lang::reference_t has a NULL baseType",
                 baseType);
      baseType->printLeft(pout);
      if (pout.lastCharNeedsWhitespace()) {
        pout << ' ';
      }
      pout << '&';
    }
    //==================================

    //---[ Structure ]------------------
    const int structure_t::class_  = (1 << 0);
    const int structure_t::enum_   = (1 << 1);
    const int structure_t::struct_ = (1 << 2);
    const int structure_t::union_  = (1 << 3);

    structure_t::structure_t(const std::string &name_,
                             const int stype_) :
      type_t(name_),
      stype(stype_) {}

    structure_t::structure_t(const qualifiers_t &qualifiers_,
                             const std::string &name_,
                             const int stype_) :
      type_t(qualifiers_, name_),
      stype(stype_) {}

    structure_t::structure_t(const std::string &name_,
                             const int stype_,
                             const blockStatement &body_) :
      type_t(name_),
      stype(stype_),
      body(body_) {}

    structure_t::structure_t(const qualifiers_t &qualifiers_,
                             const std::string &name_,
                             const int stype_,
                             const blockStatement &body_) :
      type_t(qualifiers_, name_),
      stype(stype_),
      body(body_) {}

    structure_t::~structure_t() {}

    int structure_t::type() const {
      return specifierType::class_;
    }

    type_t& structure_t::clone() const {
      return *(new structure_t(qualifiers,
                               name,
                               stype,
                               body));
    }

    bool structure_t::canBeDereferenced() const {
      return false;
    }

    bool structure_t::canBeCastedToExplicitly(const type_t &alias) const {
      // TODO: Handle class casting
      return false;
    }

    bool structure_t::canBeCastedToImplicitly(const type_t &alias) const {
      // TODO: Handle class casting
      return false;
    }

    void structure_t::printDeclaration(printer &pout) const {
      pout.printIndentation();

      switch (stype) {
      case structure_t::class_ : pout << "class" ; break;
      case structure_t::enum_  : pout << "enum"  ; break;
      case structure_t::struct_: pout << "struct"; break;
      case structure_t::union_ : pout << "union" ; break;
      }
      if (name.size()) {
        pout << ' ' << name;
      }
      if (body.size()) {
        pout.pushInlined(true);
        body.print(pout);
        pout.pushInlined(false);
      } else {
        pout << " {}";
      }
      pout << ";\n";
    }
    //==================================

    //---[ Typedef ]--------------------
    typedef_t::typedef_t(const type_t &baseType_,
                         const std::string &name_) :
      type_t(baseType_, name_) {}

    typedef_t::typedef_t(const qualifiers_t &qualifiers_,
                         const type_t &baseType_,
                         const std::string &name_) :
      type_t(qualifiers_, baseType_, name_) {}

    typedef_t::~typedef_t() {}

    int typedef_t::type() const {
      return specifierType::typedef_;
    }

    type_t& typedef_t::clone() const {
      return *(new typedef_t(qualifiers, *baseType, name));
    }

    bool typedef_t::canBeDereferenced() const {
      return baseType->canBeDereferenced();
    }

    bool typedef_t::canBeCastedToExplicitly(const type_t &alias) const {
      return baseType->canBeCastedToExplicitly(alias);
    }

    bool typedef_t::canBeCastedToImplicitly(const type_t &alias) const {
      return baseType->canBeCastedToImplicitly(alias);
    }

    void typedef_t::printLeft(printer &pout) const {
      pout << name;
    }

    void typedef_t::printDeclaration(printer &pout) const {
      OCCA_ERROR("occa::lang::typedef_t has a NULL baseType",
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

    int attribute_t::type() const {
      return specifierType::attribute;
    }

    void attribute_t::print(printer &pout) const {
      pout << '@' << name;
    }
    //==================================

    //---[ Type Checking ]--------------
    bool typesAreEqual(const type_t *a, const type_t *b) {
      qualifiers_t aQualifiers, bQualifiers;
      return typesAreEqual(aQualifiers, a,
                           bQualifiers, b);
    }

    bool typesAreEqual(qualifiers_t &aQualifiers, const type_t *a,
                       qualifiers_t &bQualifiers, const type_t *b) {
      // Check NULL case
      if (!a || !b) {
        return (a == b);
      }

      a = extractBaseTypes(aQualifiers, a);
      b = extractBaseTypes(bQualifiers, b);

      const int aType = a->type();

      // Check qualifiers, type, and existence of baseType
      if ((aQualifiers != bQualifiers) ||
          (aType != b->type())         ||
          ((!a->baseType) != (!b->baseType))) {
        return false;
      }
      if (aType & specifierType::primitive) {
        return (a == b);
      }
      if (!a->baseType) {
        return true;
      }

      // Still have types to test
      aQualifiers.clear();
      bQualifiers.clear();
      return typesAreEqual(aQualifiers, a->baseType,
                           bQualifiers, b->baseType);
    }

    const type_t* extractBaseTypes(qualifiers_t &qualifiers,
                                   const type_t *t) {
      qualifiers += t->qualifiers;

      if (t->type() & (specifierType::type |
                       specifierType::typedef_)) {
        return extractBaseTypes(qualifiers, t->baseType);
      }
      return t;
    }
    //==================================
  }
}
