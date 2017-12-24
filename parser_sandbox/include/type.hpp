#ifndef OCCA_PARSER_TYPE_HEADER2
#define OCCA_PARSER_TYPE_HEADER2

#include <ostream>
#include <vector>
#include <map>

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "printer.hpp"

// TODO: Add mangle logic here

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

      static const stype_t canBeDereferenced = (pointer |
                                                array);
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
    class qualifiers {
    public:
      qualifierVector_t qualifierVec;

      qualifiers();
      qualifiers(const qualifier &q);
      ~qualifiers();

      inline int size() const {
        return (int) qualifierVec.size();
      }

      int has(const qualifier &q);
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
      qualifiers qualifiers_;

      type_t();
      type_t(const std::string &name_);
      type_t(const type_t &baseType_,
             const std::string &name_ = "");
      type_t(const qualifiers &qs,
             const std::string &name_ = "");
      type_t(const qualifiers &qs,
             const type_t &baseType_,
             const std::string &name_ = "");

      virtual ~type_t();

      virtual stype_t type() const;

      void replaceBaseType(type_t &baseType_);

      virtual type_t& clone() const;

      inline void addQualifier(const qualifier &q) {
        qualifiers_.add(q);
      }

      inline void removeQualifier(const qualifier &q) {
        qualifiers_.remove(q);
      }

      inline bool hasQualifier(const qualifier &q) {
        return qualifiers_.has(q) >= 0;
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
    };
    //==================================

    //---[ Pointer ]--------------------
    class pointerType : public type_t {
    public:
      qualifiers rightQualifiers;

      pointerType(const type_t &baseType_);
      pointerType(const pointerType &baseType_);
      pointerType(const type_t &baseType_,
                  const qualifiers &rightQualifiers_);
      virtual ~pointerType();

      virtual stype_t type() const;

      virtual type_t& clone() const;

      virtual void printLeft(printer &pout) const;
    };
    //==================================

    //---[ Array ]----------------------
    class arrayType : public type_t {
    public:
      const exprNode *size;

      arrayType(const type_t &baseType_);
      arrayType(const arrayType &baseType_);
      arrayType(const type_t &baseType_,
                const exprNode &size_);
      virtual ~arrayType();

      virtual stype_t type() const;

      void setSize(exprNode &size_);

      virtual type_t& clone() const;

      virtual void printRight(printer &pout) const;
    };
    //==================================

    //---[ Reference ]------------------
    class referenceType : public type_t {
    public:
      referenceType(const type_t &baseType_);
      referenceType(const referenceType &baseType_);
      virtual ~referenceType();

      virtual stype_t type() const;

      virtual type_t& clone() const;

      virtual void printLeft(printer &pout) const;
    };
    //==================================

    //---[ Class ]----------------------
    class classType : public declarationType {
      std::string name;
      qtype_t label;
      blockStatement *body;

      classType(const std::string &name_,
                const int label_);

      classType(const std::string &name_,
                const int label_,
                blockStatement &body_);

      virtual stype_t type() const;

      virtual ~classType();

      virtual type_t& clone() const;

      virtual void printDeclaration(printer &pout) const;
    };
    //==================================

    //---[ Typedef ]--------------------
    class typedefType : public declarationType {
    public:
      typedefType(const type_t &baseType_,
                  const std::string &name_);
      typedefType(const qualifiers &qs,
                  const type_t &baseType_,
                  const std::string &name_);
      virtual ~typedefType();

      virtual stype_t type() const;

      virtual type_t& clone() const;

      virtual void printLeft(printer &pout) const;
      virtual void printDeclaration(printer &pout) const;
    };
    //==================================

    //---[ Function ]-------------------
    class functionType : public declarationType {
    public:
      typeVector_t args;

      functionType(const type_t &returnType);
      functionType(const functionType &returnType);
      functionType(const type_t &returnType, const std::string &name_);
      virtual ~functionType();

      virtual stype_t type() const;

      const type_t& returnType() const;
      void setReturnType(const type_t &returnType);

      void addArgument(const type_t &argType,
                       const std::string &argName = "");

      void addArgument(const qualifiers &qs,
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

    //---[ Attribute ]------------------
    class attribute : public specifier {
      attribute(const std::string &name_);
      virtual ~attribute();

      virtual stype_t type() const;

      virtual void print(printer &pout) const;
    };
    //==================================
  }
}
#endif
