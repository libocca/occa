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

#include <vector>

#include <occa/lang/attribute.hpp>
#include <occa/lang/baseStatement.hpp>
#include <occa/lang/printer.hpp>
#include <occa/lang/qualifier.hpp>

namespace occa {
  namespace lang {
    class token_t;
    class identifierToken;
    class operatorToken;
    class exprNode;
    class blockStatement;
    class pointer_t;
    class array_t;
    class variable_t;

    typedef std::vector<pointer_t>   pointerVector;
    typedef std::vector<array_t>     arrayVector;
    typedef std::vector<variable_t>  variableVector;
    typedef std::vector<variable_t*> variablePtrVector;

    namespace typeType {
      extern const int none;

      extern const int primitive;
      extern const int typedef_;
      extern const int functionPtr;
      extern const int function;

      extern const int class_;
      extern const int struct_;
      extern const int union_;
      extern const int enum_;
      extern const int structure;
    }

    namespace classAccess {
      extern const int private_;
      extern const int protected_;
      extern const int public_;
    }

    //---[ Type ]-----------------------
    class type_t {
    public:
      identifierToken *source;
      attributeTokenMap attributes;

      type_t(const std::string &name_ = "");
      type_t(const identifierToken &source_);
      type_t(const type_t &other);
      virtual ~type_t();

      virtual int type() const = 0;
      virtual type_t& clone() const = 0;

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

      template <class TM>
      inline TM& constTo() const {
        TM *ptr = dynamic_cast<TM*>(const_cast<type_t*>(this));
        OCCA_ERROR("Unable to cast type_t::to",
                   ptr != NULL);
        return *ptr;
      }

      void setSource(const identifierToken &source_);

      virtual const std::string& name() const;
      virtual bool isNamed() const;

      virtual bool isPointerType() const;

      bool operator == (const type_t &other) const;
      bool operator != (const type_t &other) const;
      virtual bool equals(const type_t &other) const;

      bool hasAttribute(const std::string &attr) const;

      virtual void printDeclaration(printer &pout) const = 0;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };

    std::ostream& operator << (std::ostream &out,
                               const type_t &type);
    printer& operator << (printer &pout,
                          const type_t &type);
    //==================================

    //---[ Pointer ]--------------------
    class pointer_t {
    public:
      qualifiers_t qualifiers;

      pointer_t();
      pointer_t(const qualifiers_t &qualifiers_);
      pointer_t(const pointer_t &other);

      bool has(const qualifier_t &qualifier) const;

      void operator += (const qualifier_t &qualifier);
      void operator -= (const qualifier_t &qualifier);
      void operator += (const qualifiers_t &others);

      void add(const fileOrigin &origin,
               const qualifier_t &qualifier);

      void add(const qualifierWithSource &qualifier);
    };

    std::ostream& operator << (std::ostream &out,
                               const pointer_t &pointer);
    printer& operator << (printer &pout,
                          const pointer_t &pointer);
    //==================================

    //---[ Array ]----------------------
    class array_t {
    public:
      operatorToken *start, *end;
      exprNode *size;

      array_t();

      array_t(const operatorToken &start_,
              const operatorToken &end_,
              exprNode *size_);

      array_t(const array_t &other);

      ~array_t();

      bool hasSize() const;
      bool canEvaluateSize() const;
      primitive evaluateSize() const;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };

    std::ostream& operator << (std::ostream &out,
                               const array_t &array);
    printer& operator << (printer &pout,
                          const array_t &array);
    //==================================

    //---[ Vartype ]--------------------
    class vartype_t {
    public:
      qualifiers_t qualifiers;

      identifierToken *typeToken;
      const type_t *type;

      pointerVector pointers;
      token_t *referenceToken;
      arrayVector arrays;

      int bitfield;

      vartype_t();

      vartype_t(const type_t &type_);

      vartype_t(const identifierToken &typeToken_,
                const type_t &type_);

      vartype_t(const vartype_t &other);

      ~vartype_t();

      vartype_t& operator = (const vartype_t &other);

      void clear();

      bool isValid() const;
      bool isNamed() const;
      std::string name() const;

      fileOrigin origin() const;

      bool isPointerType() const;

      void setReferenceToken(token_t *token);
      bool isReference() const;

      bool operator == (const vartype_t &other) const;
      bool operator != (const vartype_t &other) const;

      bool has(const qualifier_t &qualifier) const;

      vartype_t& operator += (const qualifier_t &qualifier);
      vartype_t& operator -= (const qualifier_t &qualifier);
      vartype_t& operator += (const qualifiers_t &qualifiers_);

      void add(const fileOrigin &origin,
               const qualifier_t &qualifier);

      void add(const qualifierWithSource &qualifier);

      vartype_t& operator += (const pointer_t &pointer);
      vartype_t& operator += (const pointerVector &pointers_);

      vartype_t& operator += (const array_t &array);
      vartype_t& operator += (const arrayVector &arrays_);

      bool hasAttribute(const std::string &attr) const;

      vartype_t declarationType() const;

      vartype_t flatten() const;

      void printDeclaration(printer &pout,
                            const std::string &varName,
                            const bool printType = true) const;

      void printExtraDeclaration(printer &pout,
                                 const std::string &varName) const;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };

    std::ostream& operator << (std::ostream &out,
                               const vartype_t &type);
    printer& operator << (printer &pout,
                          const vartype_t &type);
    //==================================

    //---[ Types ]----------------------
    class primitive_t : public type_t {
    public:
      const std::string pname;

      primitive_t(const std::string &name_);

      virtual const std::string& name() const;
      virtual bool isNamed() const;

      virtual int type() const;
      virtual type_t& clone() const;

      virtual void printDeclaration(printer &pout) const;
    };

    class typedef_t : public type_t {
    public:
      vartype_t baseType;

      typedef_t(const vartype_t &baseType_);

      typedef_t(const vartype_t &baseType_,
                identifierToken &source_);

      typedef_t(const typedef_t &other);

      virtual int type() const;
      virtual type_t& clone() const;

      virtual bool isPointerType() const;

      virtual bool equals(const type_t &other) const;

      virtual void printDeclaration(printer &pout) const;
    };

    class functionPtr_t : public type_t {
    public:
      vartype_t returnType;
      variableVector args;

      // Obj-C block found in OSX headers
      bool isBlock;

      functionPtr_t();

      functionPtr_t(const vartype_t &returnType_,
                    identifierToken &nameToken);

      functionPtr_t(const vartype_t &returnType_,
                    const std::string &name_ = "");

      functionPtr_t(const functionPtr_t &other);

      functionPtr_t& operator += (const variable_t &arg);
      functionPtr_t& operator += (const variableVector &args_);

      virtual int type() const;
      virtual type_t& clone() const;

      virtual bool isPointerType() const;

      virtual bool equals(const type_t &other) const;

      virtual void printDeclaration(printer &pout) const;
    };

    class function_t : public type_t {
    public:
      vartype_t returnType;
      variablePtrVector args;

      function_t();

      function_t(const vartype_t &returnType_,
                 identifierToken &nameToken);

      function_t(const vartype_t &returnType_,
                 const std::string &name_ = "");

      function_t(const function_t &other);

      void free();

      virtual int type() const;
      virtual type_t& clone() const;

      function_t& operator += (const variable_t &arg);
      function_t& operator += (const variableVector &args_);

      virtual bool equals(const type_t &other) const;

      virtual void printDeclaration(printer &pout) const;
    };

    class structure_t : public type_t {
    public:
      blockStatement body;

      structure_t(const std::string &name_ = "");
    };

    class class_t : public structure_t {
    public:
      class_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual void printDeclaration(printer &pout) const;
    };

    class struct_t : public structure_t {
    public:
      struct_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual void printDeclaration(printer &pout) const;
    };

    class enum_t : public structure_t {
    public:
      enum_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual void printDeclaration(printer &pout) const;
    };

    class union_t : public structure_t {
    public:
      union_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual void printDeclaration(printer &pout) const;
    };
    //==================================
  }
}

#endif
