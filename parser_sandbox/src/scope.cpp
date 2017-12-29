#include "scope.hpp"

namespace occa {
  namespace lang {
    scope_t::scope_t(context &ctx_) :
      ctx(ctx_) {}

    void scope_t::add(typedefType &value) {
      add(value, keywordType::typedef_);
    }

    void scope_t::add(classType &value) {
      add(value, keywordType::class_);
    }

    void scope_t::add(functionType &value) {
      add(value, keywordType::function_);
    }

    void scope_t::add(attribute &value) {
      add(value, keywordType::attribute);
    }

    void scope_t::add(specifier &value, const int ktype) {
      // Ctx checks for duplicates
      ctx.add(value, ktype);
      trie.add(value.uniqueName(),
               keyword_t(ktype, &value));
    }

    keyword_t scope_t::get(const std::string &name) {
      return trie.get(name).value();
    }
  }
}
