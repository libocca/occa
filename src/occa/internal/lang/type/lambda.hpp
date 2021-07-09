#ifndef OCCA_LANG_TYPE_LAMBDA_HEADER
#define OCCA_LANG_TYPE_LAMBDA_HEADER

#include <occa/internal/lang/type/type.hpp>
#include <occa/internal/lang/type/vartype.hpp>
#include <occa/internal/lang/expr/expr.hpp>
#include <occa/internal/lang/type/function.hpp>
// #include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    enum class capture_t { byReference, byValue };

    // For now assume all variables are captured by value or reference.
    // Add support for capturing specific variables later, if necessary.
    class lambda_t : public function_t {
    public:
      capture_t capture;
      blockStatement* body;

      lambda_t();

      lambda_t(capture_t capture_);

      lambda_t(capture_t capture_,const blockStatement& body_);

      lambda_t(const lambda_t &other);

      ~lambda_t();

      // void free();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual bool isNamed() const;
      virtual dtype_t dtype() const;

      virtual bool equals(const type_t &other) const;

      void debugPrint() const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
