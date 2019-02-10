#include <occa/lang/scope.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/type.hpp>

namespace occa {
  namespace lang {
    scope_t::scope_t() {}

    scope_t::~scope_t() {
      clear();
    }

    void scope_t::clear() {
      freeKeywords(keywords, true);
    }

    scope_t scope_t::clone() const {
      scope_t other;
      keywordMap::const_iterator it = keywords.begin();
      while (it != keywords.end()) {
        other.keywords[it->first] = it->second->clone();
        ++it;
      }
      return other;
    }

    void scope_t::swap(scope_t &other) {
      keywords.swap(other.keywords);
    }

    int scope_t::size() {
      return (int) keywords.size();
    }

    bool scope_t::has(const std::string &name) {
      return (keywords.find(name) != keywords.end());
    }

    keyword_t& scope_t::get(const std::string &name) {
      static keyword_t noKeyword;
      keywordMapIterator it = keywords.find(name);
      if (it != keywords.end()) {
        return *it->second;
      }
      return noKeyword;
    }

    bool scope_t::add(type_t &type,
                      const bool force) {
      return add<typeKeyword>(type, force);
    }

    bool scope_t::add(function_t &func,
                      const bool force) {
      return add<functionKeyword>(func, force);
    }

    bool scope_t::add(variable_t &var,
                      const bool force) {
      return add<variableKeyword>(var, force);
    }

    void scope_t::remove(const std::string &name,
                         const bool deleteSource) {
      keywordMapIterator it = keywords.find(name);
      if (it != keywords.end()) {
        keyword_t &keyword = *(it->second);
        if (deleteSource) {
          keyword.deleteSource();
        }
        delete &keyword;
        keywords.erase(it);
      }
    }

    void scope_t::moveTo(scope_t &scope) {
      scope.keywords.insert(keywords.begin(),
                            keywords.end());
      keywords.clear();
    }

    void scope_t::debugPrint() const {
      keywordMap::const_iterator it = keywords.begin();
      while (it != keywords.end()) {
        io::stdout << '['
                   << stringifySetBits(it->second->type())
                   << "] "
                   << it->first << '\n';
        ++it;
      }
    }
  }
}
