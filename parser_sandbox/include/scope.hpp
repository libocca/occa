#ifndef OCCA_PARSER_SCOPE_HEADER2
#define OCCA_PARSER_SCOPE_HEADER2

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/sys.hpp"

#include "trie.hpp"
#include "type.hpp"
#include "keyword.hpp"
#include "context.hpp"

namespace occa {
  namespace lang {
    class scope_t {
      context_t &context;
      keywordTrie trie;

    public:
      scope_t(context_t &context_);

      void add(typedefType  &value);
      void add(classType    &value);
      void add(functionType &value);
      void add(attribute    &value);
      void add(specifier &value, const int ktype);

      keyword_t get(const std::string &name);
    };
  }
}

#endif
