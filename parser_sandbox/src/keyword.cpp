#include "keyword.hpp"

namespace occa {
  namespace lang {
    keyword_t::keyword_t() :
      ktype(keywordType::none),
      ptr(NULL) {}

    keyword_t::keyword_t(const int ktype_,
                         specifier *ptr_) :
      ktype(ktype_),
      ptr(ptr_) {}
  }
}
