#ifndef OCCA_INTERNAL_LANG_ENUMERATOR_HEADER
#define OCCA_INTERNAL_LANG_ENUMERATOR_HEADER
#include <occa/dtype.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/type.hpp>
#include <occa/internal/io/output.hpp>

namespace occa {
  namespace lang {
    class exprNode;

    //---[ Enumerator ]--------------
    class enumerator_t {
    public:
      identifierToken *source;
      exprNode *expr;

      enumerator_t();
      enumerator_t(identifierToken *source, exprNode *expr_);
      enumerator_t(const std::string &name_, exprNode *expr_);
      enumerator_t(const enumerator_t &other);

      enumerator_t& operator = (const enumerator_t &other);
      enumerator_t& clone() const;

      ~enumerator_t();

      void clear();
      bool exists() const;

    };
    //==================================
  }
}

#endif
