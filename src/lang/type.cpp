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

#include <occa/lang/exprNode.hpp>
#include <occa/lang/type.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace typeType {
      const int none      = (1 << 0);

      const int primitive   = (1 << 1);
      const int typedef_    = (1 << 2);
      const int functionPtr = (1 << 3);
      const int function    = (1 << 4);

      const int class_      = (1 << 5);
      const int struct_     = (1 << 6);
      const int union_      = (1 << 7);
      const int enum_       = (1 << 8);
      const int structure   = (class_  |
                               struct_ |
                               union_  |
                               enum_);
    }

    namespace classAccess {
      const int private_   = (1 << 0);
      const int protected_ = (1 << 1);
      const int public_    = (1 << 2);
    }

    //---[ Type ]-----------------------
    type_t::type_t(const std::string &name_) :
      source(new identifierToken(fileOrigin(),
                                 name_)) {}

    type_t::type_t(const identifierToken &source_) :
      source((identifierToken*) token_t::clone(&source_)) {}

    type_t::type_t(const type_t &other) :
      source((identifierToken*) token_t::clone(other.source)),
      attributes(other.attributes) {}

    type_t::~type_t() {
      delete source;
    }

    void type_t::setSource(const identifierToken &source_) {
      if (source != &source_ ) {
        delete source;
        source = (identifierToken*) source_.clone();
      }
    }

    const std::string& type_t::name() const {
      static std::string noName;
      if (source) {
        return source->value;
      }
      return noName;
    }

    bool type_t::isNamed() const {
      if (source) {
        return source->value.size();
      }
      return false;
    }

    bool type_t::isPointerType() const {
      return false;
    }

    bool type_t::operator == (const type_t &other) const {
      if (type() != other.type()) {
        return false;
      }
      if (this == &other) {
        return true;
      }
      return equals(other);
    }

    bool type_t::operator != (const type_t &other) const {
      return !(*this == other);
    }

    bool type_t::equals(const type_t &other) const {
      return false;
    }

    bool type_t::hasAttribute(const std::string &attr) const {
      return (attributes.find(attr) != attributes.end());
    }

    void type_t::printWarning(const std::string &message) const {
      if (!source) {
        occa::printWarning(std::cerr, "[No Token] " + message);
      } else {
        source->printWarning(message);
      }
    }

    void type_t::printError(const std::string &message) const {
      if (!source) {
        occa::printError(std::cerr, "[No Token] " + message);
      } else {
        source->printError(message);
      }
    }

    std::ostream& operator << (std::ostream &out,
                               const type_t &type) {
      printer pout(out);
      pout << type;
      return out;
    }

    printer& operator << (printer &pout,
                          const type_t &type) {
      pout << type.name();
      return pout;
    }
    //==================================

    //---[ Pointer ]--------------------
    pointer_t::pointer_t() {}

    pointer_t::pointer_t(const qualifiers_t &qualifiers_) :
      qualifiers(qualifiers_) {}

    pointer_t::pointer_t(const pointer_t &other) :
      qualifiers(other.qualifiers) {}

    bool pointer_t::has(const qualifier_t &qualifier) const {
      return qualifiers.has(qualifier);
    }

    void pointer_t::operator += (const qualifier_t &qualifier) {
      qualifiers += qualifier;
    }

    void pointer_t::operator -= (const qualifier_t &qualifier) {
      qualifiers -= qualifier;
    }

    void pointer_t::operator += (const qualifiers_t &qualifiers_) {
      qualifiers += qualifiers_;
    }

    void pointer_t::add(const fileOrigin &origin,
                        const qualifier_t &qualifier) {
      qualifiers.add(origin, qualifier);
    }

    void pointer_t::add(const qualifierWithSource &qualifier) {
      qualifiers.add(qualifier);
    }

    std::ostream& operator << (std::ostream &out,
                               const pointer_t &pointer) {
      printer pout(out);
      pout << pointer;
      return out;
    }

    printer& operator << (printer &pout,
                          const pointer_t &pointer) {
      pout << '*';
      if (pointer.qualifiers.size()) {
        pout << ' ' << pointer.qualifiers;
      }
      return pout;
    }
    //==================================

    //---[ Array ]----------------------
    array_t::array_t() :
      start(NULL),
      end(NULL),
      size(NULL) {}

    array_t::array_t(const operatorToken &start_,
                     const operatorToken &end_,
                     exprNode *size_) :
      start((operatorToken*) start_.clone()),
      end((operatorToken*) end_.clone()),
      size(size_) {}

    array_t::array_t(const array_t &other) :
      start(NULL),
      end(NULL),
      size(NULL) {
      if (other.start) {
        start = (operatorToken*) other.start->clone();
      }
      if (other.end) {
        end = (operatorToken*) other.end->clone();
      }
      if (other.size) {
        size = other.size->clone();
      }
    }

    array_t::~array_t() {
      delete start;
      delete end;
      delete size;
    }

    bool array_t::hasSize() const {
      return size;
    }

    bool array_t::canEvaluateSize() const {
      return (size &&
              size->canEvaluate());
    }

    primitive array_t::evaluateSize() const {
      return (size
              ? size->evaluate()
              : primitive());
    }

    void array_t::printWarning(const std::string &message) const {
      start->printWarning(message);
    }

    void array_t::printError(const std::string &message) const {
      start->printError(message);
    }

    std::ostream& operator << (std::ostream &out,
                               const array_t &array) {
      printer pout(out);
      pout << array;
      return out;
    }

    printer& operator << (printer &pout,
                          const array_t &array) {
      if (array.size) {
        pout << '['<< (*array.size) << ']';
      } else {
        pout << "[]";
      }
      return pout;
    }
    //==================================

    //---[ Vartype ]--------------------
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

      return *this;
    }

    void vartype_t::clear() {
      qualifiers.clear();
      pointers.clear();
      arrays.clear();
      bitfield = -1;

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

    bool vartype_t::operator == (const vartype_t &other) const {
      if (!type || !other.type) {
        return false;
      }

      vartype_t flat      = flatten();
      vartype_t otherFlat = other.flatten();

      if (((*flat.type)       != (*otherFlat.type))     ||
          (flat.isReference() != otherFlat.isReference()) ||
          (flat.qualifiers    != otherFlat.qualifiers)) {
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
                                   ? otherPointerCount
                                   : pointerCount);
      const int maxPointerCount = ((pointerCount < otherPointerCount)
                                   ? pointerCount
                                   : otherPointerCount);

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

    std::ostream& operator << (std::ostream &out,
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
    //==================================

    //---[ Types ]----------------------
    primitive_t::primitive_t(const std::string &name_) :
      type_t(name_),
      pname(name_) {}

    const std::string& primitive_t::name() const {
      return pname;
    }

    bool primitive_t::isNamed() const {
      return true;
    }

    int primitive_t::type() const {
      return typeType::primitive;
    }

    type_t& primitive_t::clone() const {
      return *(const_cast<primitive_t*>(this));
    }

    void primitive_t::printDeclaration(printer &pout) const {
      pout << name();
    }

    typedef_t::typedef_t(const vartype_t &baseType_) :
      type_t(),
      baseType(baseType_) {}

    typedef_t::typedef_t(const vartype_t &baseType_,
                         identifierToken &source_) :
      type_t(source_),
      baseType(baseType_) {}

    typedef_t::typedef_t(const typedef_t &other) :
      type_t(other),
      baseType(other.baseType) {}

    int typedef_t::type() const {
      return typeType::typedef_;
    }

    type_t& typedef_t::clone() const {
      return *(new typedef_t(*this));
    }

    bool typedef_t::isPointerType() const {
      return baseType.isPointerType();
    }

    bool typedef_t::equals(const type_t &other) const {
      return (baseType == other.to<typedef_t>().baseType);
    }

    void typedef_t::printDeclaration(printer &pout) const {
      pout << "typedef ";
      baseType.printDeclaration(pout, name());
    }

    functionPtr_t::functionPtr_t() :
      type_t(),
      returnType(),
      isBlock(false) {}

    functionPtr_t::functionPtr_t(const vartype_t &returnType_,
                                 identifierToken &nameToken) :
      type_t(nameToken),
      returnType(returnType_),
      isBlock(false) {}

    functionPtr_t::functionPtr_t(const vartype_t &returnType_,
                                 const std::string &name_) :
      type_t(name_),
      returnType(returnType_),
      isBlock(false) {}

    functionPtr_t::functionPtr_t(const functionPtr_t &other) :
      type_t(other),
      returnType(other.returnType),
      args(other.args),
      isBlock(other.isBlock) {}

    int functionPtr_t::type() const {
      return typeType::functionPtr;
    }

    type_t& functionPtr_t::clone() const {
      return *(new functionPtr_t(*this));
    }

    bool functionPtr_t::isPointerType() const {
      return true;
    }

    functionPtr_t& functionPtr_t::operator += (const variable_t &arg) {
      args.push_back(arg);
      return *this;
    }

    functionPtr_t& functionPtr_t::operator += (const variableVector &args_) {
      const int count = (int) args_.size();
      for (int i = 0; i < count; ++i) {
        args.push_back(args_[i]);
      }
      return *this;
    }

    bool functionPtr_t::equals(const type_t &other) const {
      const functionPtr_t &other_ = other.to<functionPtr_t>();

      const int argSize = (int) args.size();
      if ((isBlock != other_.isBlock)   ||
          (argSize != (int) other_.args.size())) {
        return false;
      }
      if (returnType != other_.returnType) {
        return false;
      }

      for (int i = 0; i < argSize; ++i) {
        if (args[i].vartype != other_.args[i].vartype) {
          return false;
        }
      }
      return true;
    }

    void functionPtr_t::printDeclaration(printer &pout) const {
      if (!isBlock) {
        returnType.printDeclaration(pout, "(*" + name());
      } else {
        returnType.printDeclaration(pout, "(^" + name());
      }
      pout << ')';

      pout << '(';
      const std::string argIndent = pout.indentFromNewline();
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ",\n" << argIndent;
        }
        args[i].printDeclaration(pout);
      }
      pout << ')';
    }

    function_t::function_t() :
      type_t(),
      returnType() {}

    function_t::function_t(const vartype_t &returnType_,
                           identifierToken &nameToken) :
      type_t(nameToken),
      returnType(returnType_) {}

    function_t::function_t(const vartype_t &returnType_,
                           const std::string &name_) :
      type_t(name_),
      returnType(returnType_) {}

    function_t::function_t(const function_t &other) :
      type_t(other),
      returnType(other.returnType) {

      const int count = (int) other.args.size();
      for (int i = 0; i < count; ++i) {
        args.push_back(
          &(other.args[i]->clone())
        );
      }
    }

    void function_t::free() {
      const int count = (int) args.size();
      for (int i = 0; i < count; ++i) {
        delete args[i];
      }
      args.clear();
    }

    int function_t::type() const {
      return typeType::function;
    }

    type_t& function_t::clone() const {
      return *(new function_t(*this));
    }

    function_t& function_t::operator += (const variable_t &arg) {
      args.push_back(&(arg.clone()));
      return *this;
    }

    function_t& function_t::operator += (const variableVector &args_) {
      const int count = (int) args_.size();
      for (int i = 0; i < count; ++i) {
        args.push_back(&(args_[i].clone()));
      }
      return *this;
    }

    bool function_t::equals(const type_t &other) const {
      const function_t &other_ = other.to<function_t>();

      const int argSize = (int) args.size();
      if (argSize != (int) other_.args.size()) {
        return false;
      }
      if (returnType != other_.returnType) {
        return false;
      }

      for (int i = 0; i < argSize; ++i) {
        if (args[i]->vartype != other_.args[i]->vartype) {
          return false;
        }
      }
      return true;
    }

    void function_t::printDeclaration(printer &pout) const {
      returnType.printDeclaration(pout, name());

      pout << '(';
      const std::string argIndent = pout.indentFromNewline();
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ",\n" << argIndent;
        }
        args[i]->printDeclaration(pout);
      }
      pout << ')';
    }

    // TODO: Fix body source (!NULL)
    structure_t::structure_t(const std::string &name_) :
      type_t(name_),
      body(NULL, NULL) {}

    class_t::class_t() :
      structure_t("") {}

    int class_t::type() const {
      return typeType::class_;
    }

    type_t& class_t::clone() const {
      return *(new class_t());
    }

    void class_t::printDeclaration(printer &pout) const {
    }

    struct_t::struct_t() :
      structure_t("") {}

    int struct_t::type() const {
      return typeType::struct_;
    }

    type_t& struct_t::clone() const {
      return *(new struct_t());
    }

    void struct_t::printDeclaration(printer &pout) const {
    }

    enum_t::enum_t() :
      structure_t("") {}

    int enum_t::type() const {
      return typeType::enum_;
    }

    type_t& enum_t::clone() const {
      return *(new enum_t());
    }

    void enum_t::printDeclaration(printer &pout) const {
    }

    union_t::union_t() :
      structure_t("") {}

    int union_t::type() const {
      return typeType::union_;
    }

    type_t& union_t::clone() const {
      return *(new union_t());
    }

    void union_t::printDeclaration(printer &pout) const {
    }
    //==================================
  }
}
