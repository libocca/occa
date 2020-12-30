#ifndef OCCA_INTERNAL_LANG_TYPE_FUNCTIONPTR_HEADER
#define OCCA_INTERNAL_LANG_TYPE_FUNCTIONPTR_HEADER

#include <occa/internal/lang/type/type.hpp>
#include <occa/internal/lang/type/vartype.hpp>

namespace occa {
  namespace lang {
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

      void addArgument(const variable_t &arg);
      void addArguments(const variableVector &args_);

      virtual int type() const;
      virtual type_t& clone() const;

      virtual bool isPointerType() const;

      virtual dtype_t dtype() const;

      virtual bool equals(const type_t &other) const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
