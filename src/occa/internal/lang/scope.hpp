#ifndef OCCA_INTERNAL_LANG_SCOPE_HEADER
#define OCCA_INTERNAL_LANG_SCOPE_HEADER

#include <occa/internal/lang/keyword.hpp>

namespace occa {
  namespace lang {
    // Note: scope_t doesn't clone keywords
    class scope_t {
    public:
      keywordMap keywords;

      scope_t();
      ~scope_t();

      void clear();

      scope_t clone() const;

      void swap(scope_t &other);

      int size();

      bool has(const std::string &name);
      keyword_t& get(const std::string &name);

      bool add(keyword_t &keyword, const bool force = false);

      bool add(type_t &type, const bool force = false);

      bool add(function_t &func, const bool force = false);

      bool add(variable_t &var, const bool force = false);

    private:
      template <class keywordType_, class valueType>
      bool genericAdd(valueType &value, const bool force) {
        // TODO: Use unique name
        const std::string &name = value.name();
        if (!name.size()) {
          return true;
        }

        keywordMapIterator it = keywords.find(name);
        if (it == keywords.end()) {
          keywords[name] = new keywordType_(value);
          return true;
        }

        keyword_t *&keyword = it->second;
        if (force) {
          keyword->deleteSource();
          delete keyword;
          keyword = new keywordType_(value);
          return true;
        }

        value.printError("[" + name + "] is already defined");
        keyword->printError("[" + name + "] was first defined here");

        return false;
      }

    public:
      void remove(const std::string &name,
                  const bool deleteSource = true);

      void moveTo(scope_t &scope);

      void debugPrint() const;
    };
  }
}

#endif
