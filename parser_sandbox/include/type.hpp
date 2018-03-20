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
    class qualifier_t;
    class type_t;
    class attribute_t;
    class specifier;
    class variable;
    class exprNode;

    typedef std::vector<const qualifier_t*> qualifierVector_t;
    typedef std::vector<variable*>          argVector_t;
    typedef std::vector<attribute_t>        attributeVector_t;

    //---[ Types ]----------------------
    namespace specifierType {
      extern const int none;

      extern const int qualifier;
      extern const int type;
      extern const int primitive;
      extern const int pointer;
      extern const int reference;
      extern const int array;
      extern const int struct_;
      extern const int class_;
      extern const int typedef_;
      extern const int function;
      extern const int attribute;
    }

    namespace qualifierType {
      extern const int none;

      extern const int auto_;
      extern const int const_;
      extern const int constexpr_;
      extern const int restrict_;
      extern const int signed_;
      extern const int unsigned_;
      extern const int volatile_;
      extern const int register_;
      extern const int long_;
      extern const int longlong_;
      extern const int typeInfo;

      extern const int extern_;
      extern const int static_;
      extern const int thread_local_;
      extern const int globalScope;

      extern const int friend_;
      extern const int mutable_;
      extern const int classInfo;

      extern const int inline_;
      extern const int virtual_;
      extern const int explicit_;
      extern const int functionInfo;

      extern const int builtin_;
      extern const int typedef_;
      extern const int class_;
      extern const int enum_;
      extern const int struct_;
      extern const int union_;
      extern const int newType;
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

      virtual int type() const = 0;

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
        return name.size();
      }

      inline bool isUnnamed() const {
        return !name.size();
      }

      virtual std::string uniqueName() const;

      virtual void print(printer &pout) const;

      std::string toString() const;
      void debugPrint() const;
    };
    //==================================

    //---[ Qualifier ]------------------
    class qualifier_t : public specifier {
    public:
      const int ktype;

      qualifier_t(const std::string &name_,
                  const int ktype_);
      virtual ~qualifier_t();

      virtual int type() const;
    };
    //==================================

    //---[ Qualifiers ]-----------------
    class qualifiers_t {
    public:
      qualifierVector_t qualifiers;

      qualifiers_t();

      qualifiers_t(const qualifier_t &qualifier);

      ~qualifiers_t();

      void clear();

      inline int size() const {
        return (int) qualifiers.size();
      }

      const qualifier_t* operator [] (const int index);

      int indexOf(const qualifier_t &qualifier) const;
      bool has(const qualifier_t &qualifier) const;

      bool operator == (const qualifiers_t &other) const;
      bool operator != (const qualifiers_t &other) const;

      qualifiers_t& operator += (const qualifier_t &qualifier);
      qualifiers_t& operator -= (const qualifier_t &qualifier);

      qualifiers_t& operator += (const qualifiers_t &others);

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

      void setBaseType(const type_t &baseType_);

      virtual int type() const;
      virtual type_t& clone() const;
      type_t& shallowClone() const;

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

      inline type_t& operator += (const qualifier_t &qualifier) {
        qualifiers += qualifier;
        return *this;
      }

      inline type_t& operator -= (const qualifier_t &qualifier) {
        qualifiers -= qualifier;
        return *this;
      }

      inline bool has(const qualifier_t &qualifier) {
        return qualifiers.has(qualifier);
      }

      virtual void printLeft(printer &pout) const;
      virtual void printRight(printer &pout) const;

      virtual void print(printer &pout) const;
    };

    class declarationType : virtual public type_t {
    public:
      virtual ~declarationType();

      virtual void printDeclaration(printer &pout) const = 0;

      std::string declarationToString() const;
      void declarationDebugPrint() const;
    };

    void shallowFree(const type_t *&type);
    //==================================

    //---[ Primitive ]------------------
    class primitive_t : public type_t {
    public:
      primitive_t(const std::string &name_);
      virtual ~primitive_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;
    };
    //==================================

    //---[ Base Pointer ]---------------
    class basePointer_t : virtual public type_t {
    public:
      virtual ~basePointer_t();

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;
    };
    //==================================

    //---[ Pointer ]--------------------
    class pointer_t : public basePointer_t {
    public:
      pointer_t(const type_t &baseType_);
      pointer_t(const qualifiers_t &qualifiers_,
                const type_t &baseType_);

      pointer_t(const pointer_t &baseType_);

      virtual ~pointer_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual void printLeft(printer &pout) const;
    };
    //==================================

    //---[ Array ]----------------------
    class array_t : public basePointer_t {
    public:
      const exprNode *size;

      array_t(const type_t &baseType_);

      array_t(const qualifiers_t &qualifiers_,
              const type_t &baseType_);

      array_t(const type_t &baseType_,
              const exprNode &size_);

      array_t(const qualifiers_t &qualifiers_,
              const type_t &baseType_,
              const exprNode &size_);

      array_t(const array_t &baseType_);

      virtual ~array_t();

      virtual int type() const;
      virtual type_t& clone() const;

      void setSize(exprNode &size_);

      virtual void printRight(printer &pout) const;
    };
    //==================================

    //---[ Function ]-------------------
    class function_t : public declarationType,
                       public basePointer_t {
    public:
      argVector_t args;
      blockStatement body;

      // Obj-C block found in OSX headers
      bool isBlock;

      function_t(const type_t &returnType);

      function_t(const type_t &returnType,
                 const std::string &name_);

      function_t(const type_t &returnType,
                 const std::string &name_,
                 const argVector_t &args_);

      virtual ~function_t();

      virtual int type() const;
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
    class reference_t : public type_t {
    public:
      reference_t(const type_t &baseType_);

      reference_t(const qualifiers_t &qualifiers_,
                  const type_t &baseType_);

      reference_t(const reference_t &baseType_);

      virtual ~reference_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;

      virtual void printLeft(printer &pout) const;
    };
    //==================================

    //---[ Structure ]------------------
    class structure_t : public declarationType {
    public:
      std::string name;
      int stype;
      blockStatement body;

      static const int class_;
      static const int enum_;
      static const int struct_;
      static const int union_;

      structure_t(const std::string &name_,
                  const int stype_);

      structure_t(const qualifiers_t &qualifiers_,
                  const std::string &name_,
                  const int stype_);

      structure_t(const std::string &name_,
                  const int stype_,
                  const blockStatement &body_);

      structure_t(const qualifiers_t &qualifiers_,
                  const std::string &name_,
                  const int stype_,
                  const blockStatement &body_);

      virtual ~structure_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual bool canBeDereferenced() const;
      virtual bool canBeCastedToExplicitly(const type_t &alias) const;
      virtual bool canBeCastedToImplicitly(const type_t &alias) const;

      virtual void printDeclaration(printer &pout) const;
    };
    //==================================

    //---[ Typedef ]--------------------
    class typedef_t : public declarationType {
    public:
      typedef_t(const type_t &baseType_,
                const std::string &name_);

      typedef_t(const qualifiers_t &qualifiers_,
                const type_t &baseType_,
                const std::string &name_);

      virtual ~typedef_t();

      virtual int type() const;
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

      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Type Checking ]--------------
    bool typesAreEqual(const type_t *a, const type_t *b);

    bool typesAreEqual(qualifiers_t &aQualifiers, const type_t *a,
                       qualifiers_t &bQualifiers, const type_t *b);

    const type_t* extractBaseTypes(qualifiers_t &qualifiers,
                                   const type_t *t);
    //==================================
  }
}
#endif
