#ifndef OCCA_INTERNAL_LANG_TYPE_ARRAY_HEADER
#define OCCA_INTERNAL_LANG_TYPE_ARRAY_HEADER

#include <occa/internal/lang/type/type.hpp>

namespace occa {
  namespace lang {
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

    io::output& operator << (io::output &out,
                             const array_t &array);

    printer& operator << (printer &pout,
                          const array_t &array);
  }
}

#endif
