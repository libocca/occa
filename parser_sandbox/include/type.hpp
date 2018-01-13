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
#ifndef OCCA_PARSER_TYPE_HEADER2
#define OCCA_PARSER_TYPE_HEADER2

#include <ostream>
#include <vector>
#include <map>

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/sys.hpp"
#include "printer.hpp"

// TODO: Add mangle logic to uniqueName()

namespace occa {
  namespace lang {
    class qualifier;
    class type_t;
    class variable;
    class exprNode;
    class blockStatement;

    typedef std::vector<const qualifier*> qualifierVector_t;
    typedef std::vector<type_t*> typeVector_t;

    typedef int      stype_t;
    typedef uint64_t qtype_t;

    class specifierType {
    public:
      static const stype_t none = 0;

      static const stype_t qualifier = (1 << 0);
      static const stype_t type      = (1 << 1);
      static const stype_t primitive = (1 << 2);
      static const stype_t pointer   = (1 << 3);
      static const stype_t reference = (1 << 4);
      static const stype_t array     = (1 << 5);
      static const stype_t class_    = (1 << 6);
      static const stype_t typedef_  = (1 << 7);
      static const stype_t function  = (1 << 8);
      static const stype_t attribute = (1 << 9);

      static const stype_t printsOnBothSides = (array |
                                                function);
    };

    class qualifierType {
    public:
      static const qtype_t none          = 0;

      static const qtype_t auto_         = (1L << 0);
      static const qtype_t const_        = (1L << 1);
      static const qtype_t constexpr_    = (1L << 2);
      static const qtype_t restrict_     = (1L << 3);
      static const qtype_t signed_       = (1L << 4);
      static const qtype_t unsigned_     = (1L << 5);
      static const qtype_t volatile_     = (1L << 6);
      static const qtype_t register_     = (1L << 7);
      static const qtype_t typeInfo      = (const_     |
                                            constexpr_ |
                                            signed_    |
                                            unsigned_  |
                                            volatile_  |
                                            register_);

      static const qtype_t extern_       = (1L << 8);
      static const qtype_t static_       = (1L << 9);
      static const qtype_t thread_local_ = (1L << 10);
      static const qtype_t globalScope   = (extern_ |
                                            static_ |
                                            thread_local_);

      static const qtype_t friend_       = (1L << 11);
      static const qtype_t mutable_      = (1L << 12);
      static const qtype_t classInfo     = (friend_ |
                                            mutable_);

      static const qtype_t inline_       = (1L << 13);
      static const qtype_t virtual_      = (1L << 14);
      static const qtype_t explicit_     = (1L << 15);
      static const qtype_t functionInfo  = (typeInfo |
                                            inline_  |
                                            virtual_ |
                                            explicit_);

      static const qtype_t builtin_      = (1L << 16);
      static const qtype_t typedef_      = (1L << 17);
      static const qtype_t class_        = (1L << 18);
      static const qtype_t enum_         = (1L << 19);
      static const qtype_t struct_       = (1L << 20);
      static const qtype_t union_        = (1L << 21);
      static const qtype_t newType       = (typedef_ |
                                            class_   |
                                            enum_    |
                                            struct_  |
                                            union_);
    };

    class classLabel {
    public:
      static const int class_  = 1;
      static const int enum_   = 2;
      static const int struct_ = 3;
      static const int union_  = 4;
    };

    class classAccess {
    public:
      static const int private_   = (1 << 0);
      static const int protected_ = (1 << 1);
      static const int public_    = (1 << 2);
    };

    //---[ Specifier ]------------------
    class specifier {
    public:
      std::string name;

      specifier();
      specifier(const std::string &name_);
      virtual ~specifier();

      virtual stype_t type() const = 0;

      inline bool isNamed() const {
        return (name.size() != 0);
      }

      inline bool isUnnamed() const {
        return (name.size() == 0);
      }

      virtual std::string uniqueName() const;

      virtual void print(printer &pout) const;

      std::string toString() const;
      void debugPrint() const;
    };
    //==================================

    //---[ Qualifier ]------------------
    class qualifier : public specifier {
    public:
      const qtype_t qtype;

      qualifier(const std::string &name_,
                const qtype_t qtype_);
      virtual ~qualifier();

      virtual stype_t type() const;
    };
    //==================================

    //---[ Qualifiers ]-----------------
    class qualifiers_t {
    public:
      qualifierVector_t qualifiers;

      qualifiers_t();
      qualifiers_t(const qualifier &q);
      ~qualifiers_t();

      inline int size() const {
        return (int) qualifiers.size();
      }

      int indexOf(const qualifier &q) const;
      bool has(const qualifier &q) const;

      void add(const qualifier &q);
      void remove(const qualifier &q);

      void print(printer &pout) const;

      std::string toString() const;
      void debugPrint() const;
    };
    //==================================

    //---[ Type ]-----------------------
    class type_t : public specifier {
    public:
      const type_t *baseType;
      qualifiers_t qualifiers;

      type_t();
      type_t(const std::string &name_);
      type_t(const type_t &baseType_,
             const std::string &name_ = "");
      type_t(const qualifiers_t &qualifiers_,
             const std::string &name_ = "");
      type_t(const qualifiers_t &qualifiers_,
             const type_t &baseType_,
             const std::string &name_ = "");

      virtual ~type_t();

      virtual stype_t type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast type_t::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast type_t::to",
                   ptr != NULL);
        return *ptr;
      }

      void replaceBaseType(type_t &baseType_);

      inline void addQualifier(const qualifier &q) {
        qualifiers.add(q);
      }

      inline void removeQualifier(const qualifier &q) {
        qualifiers.remove(q);
      }

      inline bool hasQualifier(const qualifier &q) {
        return qualifiers.has(q);
      }

      virtual void printLeft(printer &pout) const;
      virtual void printRight(printer &pout) const;

      virtual void print(printer &pout) const;
      void print(printer &pout, const variable &var) const;
    };

    class declarationType : public virtual type_t {
    public:
      virtual void printDeclaration(printer &pout) const = 0;

      std::string declarationToString() const;
      void declarationDebugPrint() const;
    };
    //==================================

    //---[ PrimitiveType ]--------------
    class primitiveType : public type_t {
    public:
      primitiveType(const std::string &name_);
      virtual ~primitiveType();

      virtual stype_t type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;
    };
    //==================================

    //---[ Base Pointer ]---------------
    class basePointerType : virtual public type_t {
    public:
      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;
    };
    //==================================

    //---[ Pointer ]--------------------
    class pointerType : public basePointerType {
    public:
      pointerType(const type_t &baseType_);
      pointerType(const qualifiers_t &qualifiers_,
                  const type_t &baseType_);

      pointerType(const pointerType &baseType_);

      virtual ~pointerType();

      virtual stype_t type() const;
      virtual type_t& clone() const;

      virtual void printLeft(printer &pout) const;
    };
    //==================================

    //---[ Array ]----------------------
    class arrayType : public basePointerType {
    public:
      const exprNode *size;

      arrayType(const type_t &baseType_);

      arrayType(const qualifiers_t &qualifiers_,
                const type_t &baseType_);

      arrayType(const type_t &baseType_,
                const exprNode &size_);

      arrayType(const qualifiers_t &qualifiers_,
                const type_t &baseType_,
                const exprNode &size_);

      arrayType(const arrayType &baseType_);

      virtual ~arrayType();

      virtual stype_t type() const;
      virtual type_t& clone() const;

      void setSize(exprNode &size_);

      virtual void printRight(printer &pout) const;
    };
    //==================================

    //---[ Function ]-------------------
    class functionType : public declarationType,
                         public basePointerType {
    public:
      typeVector_t args;

      functionType(const type_t &returnType);
      functionType(const type_t &returnType,
                   const std::string &name_);

      functionType(const functionType &returnType);

      virtual ~functionType();

      virtual stype_t type() const;
      virtual type_t& clone() const;

      virtual bool canBeCastedToImplicitly(const type_t &alias) const;

      const type_t& returnType() const;
      void setReturnType(const type_t &returnType);

      void addArgument(const type_t &argType,
                       const std::string &argName = "");

      void addArgument(const qualifiers_t &qualifiers_,
                       const type_t &argType,
                       const std::string &argName = "");

      int argumentCount() const {
        return (int) args.size();
      }

      void printDeclarationLeft(printer &pout) const;
      void printDeclarationRight(printer &pout) const;

      virtual void printDeclaration(printer &pout) const;
    };
    //==================================

    //---[ Reference ]------------------
    class referenceType : public type_t {
    public:
      referenceType(const type_t &baseType_);
      referenceType(const qualifiers_t &qualifiers_,
                    const type_t &baseType_);
      referenceType(const referenceType &baseType_);
      virtual ~referenceType();

      virtual stype_t type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;

      virtual void printLeft(printer &pout) const;
    };
    //==================================

    //---[ Class ]----------------------
    class classType : public declarationType {
    public:
      std::string name;
      qtype_t label;
      blockStatement *body;

      classType(const std::string &name_,
                const int label_);
      classType(const qualifiers_t &qualifiers_,
                const std::string &name_,
                const int label_);

      classType(const std::string &name_,
                const int label_,
                blockStatement &body_);
      classType(const qualifiers_t &qualifiers_,
                const std::string &name_,
                const int label_,
                blockStatement &body_);

      virtual ~classType();

      virtual stype_t type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;

      virtual void printDeclaration(printer &pout) const;
    };
    //==================================

    //---[ Typedef ]--------------------
    class typedefType : public declarationType {
    public:
      typedefType(const type_t &baseType_,
                  const std::string &name_);
      typedefType(const qualifiers_t &qualifiers_,
                  const type_t &baseType_,
                  const std::string &name_);

      virtual ~typedefType();

      virtual stype_t type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;

      virtual void printLeft(printer &pout) const;
      virtual void printDeclaration(printer &pout) const;
    };
    //==================================

    //---[ Attribute ]------------------
    class attribute : public specifier {
    public:
      attribute(const std::string &name_);
      virtual ~attribute();

      virtual stype_t type() const;

      virtual void print(printer &pout) const;
    };
    //==================================
  }
}
#endif
