#ifndef OCCA_LANG_TYPE_LAMBDA_HEADER
#define OCCA_LANG_TYPE_LAMBDA_HEADER

#include <occa/lang/type/type.hpp>
#include <occa/lang/expr/expr.hpp>
#include <occa/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    enum class capture_t { byReference, byValue };
    class lambda_t : public type_t {
    public:
      capture_t capture;
      exprNodeVector captureParams;
      exprNodeVector args;
      blockStatement *body;

      lambda_t();

      lambda_t(capture_t cap = capture_t::byReference,
                 const std::string &name_ = "");

      lambda_t(const lambda_t &other);

      void free();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual dtype_t dtype() const;

      void addArgument(const  &arg);
      void addArguments(const variableVector &args_);

      variable_t* removeArgument(const int index);

      virtual bool equals(const type_t &other) const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
