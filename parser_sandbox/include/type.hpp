#ifndef OCCA_PARSER_TYPE_HEADER2
#define OCCA_PARSER_TYPE_HEADER2

#include <ostream>
#include <vector>
#include <map>

#include "occa/defines.hpp"
#include "occa/types.hpp"

namespace occa {
  namespace lang {
    class qualifier;
    class type;

    typedef std::vector<const qualifier*> qualifierVec_t;
    typedef std::vector<type*> typeVec_t;

    int charsFromNewline(const std::string &s);

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

      specifier(const int specType_);
      specifier(const std::string &name_, const int specType_);
      virtual ~specifier();

      inline bool isNamed() const {
        return (name.size() != 0);
      }

      inline bool isUnnamed() const {
        return (name.size() == 0);
      }

      inline std::string toString() const {
        std::string s;
        printOn(s);
        return s;
      }

      virtual void printOn(std::string &out) const;
    };

    std::ostream& operator << (std::ostream &out, const specifier &s);

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

      std::string toString() const;

      void printOn(std::string &out) const;
    };

    std::ostream& operator << (std::ostream &out, const qualifiers &qs);

    //---[ Type ]-----------------------
    class type : public specifier {
    public:
      const type *baseType;
      qualifiers qualifiers_;

      type();
      type(const std::string &name_);
      type(const std::string &name_, const int specType_);

      type(const type &baseType_, const std::string &name_ = "");
      type(const qualifiers &qs, const std::string &name_ = "");
      type(const qualifiers &qs, const type &baseType_, const std::string &name_ = "");

      virtual ~type();

      void replaceBaseType(const type &baseType_);

      virtual type& clone() const;

      inline void add(const qualifier &q) {
        qualifiers_.add(q);
      }

      inline void remove(const qualifier &q) {
        qualifiers_.remove(q);
      }

      inline bool has(const qualifier &q) {
        return qualifiers_.has(q) >= 0;
      }

      virtual void printDeclarationOn(std::string &out) const;
      virtual void printOn(std::string &out) const;
    };

    //---[ PrimitiveType ]------------------
    class primitiveType : public type {
    public:
      primitiveType(const std::string &name_);
      virtual ~primitiveType();

      virtual type& clone() const;

      virtual void printDeclarationOn(std::string &out) const;
      virtual void printOn(std::string &out) const;
    };

    //---[ Pointer ]--------------------
    class pointerType : public type {
    public:
      pointerType(const type &t);
      pointerType(const qualifiers &qs, const type &t);
      virtual ~pointerType();

      virtual type& clone() const;

      virtual void printDeclarationOn(std::string &out) const;
      virtual void printOn(std::string &out) const;
    };

    //---[ Reference ]------------------
    class referenceType : public type {
    public:
      referenceType(const type &t);
      virtual ~referenceType();

      virtual type& clone() const;

      virtual void printDeclarationOn(std::string &out) const;
      virtual void printOn(std::string &out) const;
    };

    //---[ Typedef ]--------------------
    class typedefType : public type {
    public:
      typedefType(const type &t, const std::string &name_);
      typedefType(const qualifiers &qs, const type &t, const std::string &name_);
      virtual ~typedefType();

      virtual type& clone() const;

      virtual void printDeclarationOn(std::string &out) const;
      virtual void printOn(std::string &out) const;
    };

    //---[ Class ]----------------------
    class classType : public type {
      virtual ~classType();

      virtual type& clone() const;
    };

    //---[ Function ]-------------------
    class functionType : public type {
    public:
      typeVec_t args;

      functionType(const type &returnType);
      functionType(const type &returnType, const std::string &name_);
      virtual ~functionType();

      void setReturnType(const type &returnType);
      const type& returnType() const;

      void add(const type &argType,
               const std::string &argName = "");

      void add(const qualifiers &qs,
               const type &argType,
               const std::string &argName = "");

      inline int argumentCount() const {
        return (int) args.size();
      }

      virtual void printDeclarationOn(std::string &out) const;
      virtual void printOn(std::string &out) const;
    };
  }
}
#endif
