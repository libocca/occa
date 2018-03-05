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
#ifndef OCCA_LANG_TYPE_HEADER
#define OCCA_LANG_TYPE_HEADER

#include <ostream>
#include <vector>
#include <map>

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/sys.hpp"
#include "baseStatement.hpp"
#include "printer.hpp"
#include "trie.hpp"

// TODO: Add mangle logic to uniqueName()

namespace occa {
  namespace lang {
    class qualifier;
    class type_t;
    class attribute_t;
    class specifier;
    class variable;
    class exprNode;

    typedef std::vector<const qualifier*> qualifierVector_t;
    typedef std::vector<type_t*>          typeVector_t;
    typedef std::vector<attribute_t>      attributeVector_t;
    typedef trie<const specifier*>        specifierTrie;

    typedef int ktype_t;

    //---[ Types ]----------------------
    namespace specifierType {
      extern const ktype_t none;

      extern const ktype_t qualifier;
      extern const ktype_t type;
      extern const ktype_t primitive;
      extern const ktype_t pointer;
      extern const ktype_t reference;
      extern const ktype_t array;
      extern const ktype_t struct_;
      extern const ktype_t class_;
      extern const ktype_t typedef_;
      extern const ktype_t function;
      extern const ktype_t attribute;
    }

    namespace qualifierType {
      extern const ktype_t none;

      extern const ktype_t auto_;
      extern const ktype_t const_;
      extern const ktype_t constexpr_;
      extern const ktype_t restrict_;
      extern const ktype_t signed_;
      extern const ktype_t unsigned_;
      extern const ktype_t volatile_;
      extern const ktype_t register_;
      extern const ktype_t typeInfo;

      extern const ktype_t extern_;
      extern const ktype_t static_;
      extern const ktype_t thread_local_;
      extern const ktype_t globalScope;

      extern const ktype_t friend_;
      extern const ktype_t mutable_;
      extern const ktype_t classInfo;

      extern const ktype_t inline_;
      extern const ktype_t virtual_;
      extern const ktype_t explicit_;
      extern const ktype_t functionInfo;

      extern const ktype_t builtin_;
      extern const ktype_t typedef_;
      extern const ktype_t class_;
      extern const ktype_t enum_;
      extern const ktype_t struct_;
      extern const ktype_t union_;
      extern const ktype_t newType;
    }

    namespace classAccess {
      extern const int private_;
      extern const int protected_;
      extern const int public_;
    }
    //==================================

    //---[ Specifier ]------------------
    class specifier {
    public:
      std::string name;

      specifier();
      specifier(const std::string &name_);
      virtual ~specifier();

      virtual ktype_t type() const = 0;

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast specifier::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast specifier::to",
                   ptr != NULL);
        return *ptr;
      }

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
      const ktype_t ktype;

      qualifier(const std::string &name_,
                const ktype_t ktype_);
      virtual ~qualifier();

      virtual ktype_t type() const;
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

      virtual ktype_t type() const;
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

    class declarationType : virtual public type_t {
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

      virtual ktype_t type() const;
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

      virtual ktype_t type() const;
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

      virtual ktype_t type() const;
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
      blockStatement body;

      functionType(const type_t &returnType);
      functionType(const type_t &returnType,
                   const std::string &name_);

      functionType(const functionType &returnType);

      virtual ~functionType();

      virtual ktype_t type() const;
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

      virtual ktype_t type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;

      virtual void printLeft(printer &pout) const;
    };
    //==================================

    //---[ Structure ]------------------
    class structureType : public declarationType {
    public:
      std::string name;
      int stype;
      blockStatement body;

      static const int class_;
      static const int enum_;
      static const int struct_;
      static const int union_;

      structureType(const std::string &name_,
                    const int stype_);

      structureType(const qualifiers_t &qualifiers_,
                    const std::string &name_,
                    const int stype_);

      structureType(const std::string &name_,
                    const int stype_,
                    const blockStatement &body_);

      structureType(const qualifiers_t &qualifiers_,
                    const std::string &name_,
                    const int stype_,
                    const blockStatement &body_);

      virtual ~structureType();

      virtual ktype_t type() const;
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

      virtual ktype_t type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;

      virtual void printLeft(printer &pout) const;
      virtual void printDeclaration(printer &pout) const;
    };
    //==================================

    //---[ Attribute ]------------------
    class attribute_t : public specifier {
    public:
      attribute_t(const std::string &name_);
      virtual ~attribute_t();

      virtual ktype_t type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    void getSpecifiers(specifierTrie &specifiers);
  }
}
#endif
