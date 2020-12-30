#ifndef OCCA_INTERNAL_LANG_TYPE_FUNCTION_HEADER
#define OCCA_INTERNAL_LANG_TYPE_FUNCTION_HEADER

#include <occa/internal/lang/type/type.hpp>
#include <occa/internal/lang/type/vartype.hpp>

namespace occa {
  namespace lang {
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

      virtual dtype_t dtype() const;

      void addArgument(const variable_t &arg);
      void addArguments(const variableVector &args_);

      variable_t* removeArgument(const int index);

      virtual bool equals(const type_t &other) const;

      void debugPrint() const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
