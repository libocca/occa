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

#include "exprNode.hpp"
#include "type.hpp"
#include "variable.hpp"

namespace occa {
  namespace lang {
    namespace typeType {
      const int none      = (1 << 0);

      const int primitive = (1 << 1);
      const int typedef_  = (1 << 2);
      const int function  = (1 << 3);

      const int class_    = (1 << 4);
      const int struct_   = (1 << 5);
      const int union_    = (1 << 6);
      const int enum_     = (1 << 7);
      const int structure = (class_  |
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
      name(name_) {}

    type_t::~type_t() {}

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

    printer& operator << (printer &pout,
                          const type_t &type) {
      pout << type.name;
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
    array_t::array_t(exprNode *size_) :
      size(size_) {}

    array_t::array_t(const array_t &other) :
      size(NULL) {
      if (other.size) {
        size = &(other.size->clone());
      }
    }

    array_t::~array_t() {
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
      type(NULL),
      isReference(false) {}

    vartype_t::vartype_t(const type_t &type_) :
      isReference(false) {
      if (!type_.name.size()) {
        type = &(type_.clone());
      } else {
        type = &type_;
      }
    }

    vartype_t::vartype_t(const vartype_t &other) :
      type(NULL),
      isReference(false) {
      *this = other;
    }

    vartype_t::~vartype_t() {
      clear();
    }

    vartype_t& vartype_t::operator = (const vartype_t &other) {
      clear();
      qualifiers  = other.qualifiers;
      pointers    = other.pointers;
      isReference = other.isReference;
      arrays      = other.arrays;

      if (other.type &&
          !other.type->name.size()) {
        type = &(other.type->clone());
      } else {
        type = other.type;
      }

      return *this;
    }

    void vartype_t::clear() {
      qualifiers.clear();
      pointers.clear();
      isReference = false;
      arrays.clear();

      if (type &&
          !type->name.size()) {
        delete type;
      }
      type = NULL;
    }

    bool vartype_t::isValid() const {
      return type;
    }

    bool vartype_t::operator == (const vartype_t &other) const {
      if (!type || !other.type) {
        return false;
      }

      vartype_t flat      = flatten();
      vartype_t otherFlat = other.flatten();

      if (((*flat.type)     != (*otherFlat.type))     ||
          (flat.isReference != otherFlat.isReference) ||
          (flat.qualifiers  != otherFlat.qualifiers)) {
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

    void vartype_t::operator += (const qualifier_t &qualifier) {
      qualifiers += qualifier;
    }

    void vartype_t::operator -= (const qualifier_t &qualifier) {
      qualifiers -= qualifier;
    }

    void vartype_t::operator += (const qualifiers_t &qualifiers_) {
      qualifiers += qualifiers_;
    }

    void vartype_t::operator += (const pointer_t &pointer) {
      pointers.push_back(pointer);
    }

    void vartype_t::operator += (const pointerVector &pointers_) {
      const int pointerCount = (int) pointers_.size();
      for (int i = 0; i < pointerCount; ++i) {
        pointers.push_back(pointers_[i]);
      }
    }

    void vartype_t::operator += (const array_t &array) {
      arrays.push_back(array);
    }

    void vartype_t::operator += (const arrayVector &arrays_) {
      const int arrayCount = (int) arrays_.size();
      for (int i = 0; i < arrayCount; ++i) {
        arrays.push_back(arrays_[i]);
      }
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

      return flat;
    }

    void vartype_t::printDeclaration(printer &pout,
                                     const std::string &name,
                                     const bool printType) const {
      if (!type) {
        return;
      }

      if (printType) {
        if (qualifiers.size()) {
          pout << qualifiers << ' ';
        }
        pout << *type << ' ';
      }

      const int pointerCount = (int) pointers.size();
      for (int i = 0; i < pointerCount; ++i) {
        pout << pointers[i];
        // Don't add a space after the last * if possible
        if (pointers[i].qualifiers.size()) {
          pout << ' ';
        }
      }

      pout << name;

      const int arrayCount = (int) arrays.size();
      for (int i = 0; i < arrayCount; ++i) {
        pout << arrays[i];
      }
    }

    void vartype_t::printExtraDeclaration(printer &pout,
                                          const std::string &name) const {
      printDeclaration(pout, name, false);
    }
    //==================================

    //---[ Types ]----------------------
    primitive_t::primitive_t(const std::string &name_) :
      type_t(name_) {}

    int primitive_t::type() const {
      return typeType::primitive;
    }

    type_t& primitive_t::clone() const {
      return *(const_cast<primitive_t*>(this));
    }

    void primitive_t::printDeclaration(printer &pout) const {
      pout << name;
    }

    typedef_t::typedef_t(const vartype_t &baseType_,
                         const std::string &name_) :
      type_t(name_),
      baseType(baseType_) {}

    int typedef_t::type() const {
      return typeType::typedef_;
    }

    type_t& typedef_t::clone() const {
      if (name.size()) {
        return *(const_cast<typedef_t*>(this));
      }
      return *(new typedef_t(baseType, name));
    }

    bool typedef_t::equals(const type_t &other) const {
      return (baseType == other.to<typedef_t>().baseType);
    }

    void typedef_t::printDeclaration(printer &pout) const {
      pout << "typedef ";
      baseType.printDeclaration(pout, name);
    }

    function_t::function_t() :
      type_t(""),
      returnType(),
      isPointer(false),
      isBlock(false) {}

    function_t::function_t(const vartype_t &returnType_,
                           const std::string &name_) :
      type_t(name_),
      returnType(returnType_),
      isPointer(false),
      isBlock(false) {}

    function_t::function_t(const function_t &other) :
      type_t(other.name),
      returnType(other.returnType),
      args(other.args),
      isPointer(other.isPointer),
      isBlock(other.isBlock) {}

    int function_t::type() const {
      return typeType::function;
    }

    type_t& function_t::clone() const {
      if (name.size()) {
        return *(const_cast<function_t*>(this));
      }
      return *(new function_t(*this));
    }

    function_t& function_t::operator += (const variable &arg) {
      args.push_back(arg);
      return *this;
    }

    function_t& function_t::operator += (const variableVector &args_) {
      const int count = (int) args_.size();
      for (int i = 0; i < count; ++i) {
        args.push_back(args_[i]);
      }
      return *this;
    }

    bool function_t::equals(const type_t &other) const {
      const function_t &other_ = other.to<function_t>();

      const int argSize = (int) args.size();
      if ((isPointer != other_.isPointer) ||
          (isBlock   != other_.isBlock)   ||
          (argSize   != (int) other_.args.size())) {
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

    void function_t::printDeclaration(printer &pout) const {
      const bool isPointerType = (isPointer || isBlock);
      if (!isPointerType) {
        returnType.printDeclaration(pout, name);
      } else if (isPointer) {
        returnType.printDeclaration(pout, "(*" + name);
      } else {
        returnType.printDeclaration(pout, "(^" + name);
      }
      if (isPointerType) {
        pout << ')';
      }

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

    structure_t::structure_t(const std::string &name_) :
      type_t(name_) {}

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
