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
    class statement_t;
    class qualifier;
    class type_t;

    typedef std::vector<const qualifier*> qualifierVec_t;
    typedef std::vector<type_t*> typeVec_t;

    class classAccess {
    public:
      static const int private_   = (1 << 0);
      static const int protected_ = (1 << 1);
      static const int public_    = (1 << 2);
    };

    class classLabel {
    public:
      static const int class_  = (1 << 0);
      static const int struct_ = (1 << 1);
      static const int union_  = (1 << 2);
    };

    //---[ Specifier ]------------------
    class specifier {
    public:
      std::string name;
      int specType;

      static const int qualifierType = (1 << 0);
      static const int functionType  = (1 << 1);
      static const int storageType   = (1 << 2);
      static const int variableType  = (1 << 3);
      static const int primitiveType = (1 << 4);
      static const int definedType   = (1 << 5);
      static const int attributeType = (1 << 6);

      specifier(const int specType_);
      specifier(const std::string &name_, const int specType_);
      virtual ~specifier();

      inline bool isNamed() const {
        return (name.size() != 0);
      }

      inline bool isUnnamed() const {
        return (name.size() == 0);
      }

      virtual std::string uniqueName() const;

      virtual void print(printer_t &pout) const;

      std::string toString() const;
      void debugPrint() const;
    };

    //---[ Qualifier ]------------------
    class qualifier : public specifier {
    public:
      qualifier(const std::string &name_);
      qualifier(const std::string &name_, const int specType_);
      virtual ~qualifier();
    };

    //---[ Qualifiers ]-----------------
    class qualifiers {
    public:
      qualifierVec_t qualifierVec;

      qualifiers();
      qualifiers(const qualifier &q);
      ~qualifiers();

      inline int size() const {
        return (int) qualifierVec.size();
      }

      int has(const qualifier &q);
      void add(const qualifier &q);
      void remove(const qualifier &q);

      void print(printer_t &pout) const;

      std::string toString() const;
      void debugPrint() const;
    };

    //---[ Type ]-----------------------
    class type_t : public specifier {
    public:
      const type_t *baseType;
      qualifiers qualifiers_;

      type_t();
      type_t(const std::string &name_);
      type_t(const std::string &name_, const int specType_);

      type_t(const type_t &baseType_,
             const std::string &name_ = "");
      type_t(const qualifiers &qs,
             const std::string &name_ = "");
      type_t(const qualifiers &qs,
             const type_t &baseType_,
             const std::string &name_ = "");

      virtual ~type_t();

      void replaceBaseType(const type_t &baseType_);

      virtual type_t& clone() const;

      inline void add(const qualifier &q) {
        qualifiers_.add(q);
      }

      inline void remove(const qualifier &q) {
        qualifiers_.remove(q);
      }

      inline bool has(const qualifier &q) {
        return qualifiers_.has(q) >= 0;
      }

      virtual void print(printer_t &pout) const;
    };

    class declarationType_t : public virtual type_t {
    public:
      virtual void printDeclaration(printer_t &pout) const = 0;
    };

    //---[ PrimitiveType ]------------------
    class primitiveType : public type_t {
    public:
      primitiveType(const std::string &name_);
      virtual ~primitiveType();

      virtual type_t& clone() const;

      virtual void print(printer_t &pout) const;
    };

    //---[ Pointer ]--------------------
    class pointerType : public type_t {
    public:
      pointerType(const type_t &t);
      pointerType(const qualifiers &qs, const type_t &t);
      virtual ~pointerType();

      virtual type_t& clone() const;

      virtual void print(printer_t &pout) const;
    };

    //---[ Reference ]------------------
    class referenceType : public type_t {
    public:
      referenceType(const type_t &t);
      virtual ~referenceType();

      virtual type_t& clone() const;

      virtual void print(printer_t &pout) const;
    };

    //---[ Class ]----------------------
    class classType : public declarationType_t {
      std::string name;
      int label;
      statement_t *body;

      classType(const std::string &name_,
                const int label_);

      void setBody(statement_t &body_);

      virtual ~classType();

      virtual type_t& clone() const;

      virtual void printDeclaration(printer_t &pout) const;
      virtual void print(printer_t &pout) const;
    };

    //---[ Typedef ]--------------------
    class typedefType : public declarationType_t {
    public:
      typedefType(const type_t &t, const std::string &name_);
      typedefType(const qualifiers &qs, const type_t &t, const std::string &name_);
      virtual ~typedefType();

      virtual type_t& clone() const;

      virtual void printDeclaration(printer_t &pout) const;
      virtual void print(printer_t &pout) const;
    };

    //---[ Function ]-------------------
    class functionType : public declarationType_t {
    public:
      typeVec_t args;
      std::vector<void*> defaultValues;
      mutable statement_t *body;

      functionType(const type_t &returnType);
      functionType(const type_t &returnType, const std::string &name_);
      virtual ~functionType();

      void setReturnType(const type_t &returnType);
      const type_t& returnType() const;

      void addArg(const type_t &argType,
                  const std::string &argName = "",
                  const void *defaultValue = NULL); // TODO: default values

      void addArg(const qualifiers &qs,
                  const type_t &argType,
                  const std::string &argName = "",
                  const void *defaultValue = NULL);

      inline int argumentCount() const {
        return (int) args.size();
      }

      void setBody(statement_t &body_);

      virtual void printDeclaration(printer_t &pout) const;
      virtual void print(printer_t &pout) const;
    };

    //---[ Attribute ]----------------
    class attribute : public specifier {
      attribute(const std::string &name_);
      virtual ~attribute();

      virtual void print(printer_t &pout) const;
    };
  }
}
#endif
