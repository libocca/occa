#include <algorithm>

#include "context.hpp"

namespace occa {
  namespace lang {
    statementTrie& context_t::getTrie(const int ktype) {
      switch (ktype) {
      case keywordType::qualifier: {
        return qualifierTrie;
      }
      case keywordType::primitive: {
        return primitiveTrie;
      }
      case keywordType::typedef_: {
        return typedefTrie;
      }
      case keywordType::class_: {
        return classTrie;
      }
      case keywordType::function_: {
        return functionTrie;
      }
      case keywordType::attribute: {
        return attributeTrie;
      }
      default:
        // TODO: Add more information to the error message
        OCCA_FORCE_ERROR("Keyword is not defined");
      }
      return qualifierTrie;
    }

    int context_t::getKeywordType(const std::string &name) {
      keywordTrie::result_t result = keywordMap.get(name);
      return (result.success()
              ? result.value().ktype
              : keywordType::none);
    }

    statementKeywordMap& context_t::getKeywordStatements(keyword_t &keyword) {
      return (getTrie(keyword.ktype)
              .get(keyword.ptr->uniqueName())
              .value());
    }

    void context_t::add(qualifier &value) {
      add(qualifierTrie, value, keywordType::qualifier);
    }

    void context_t::add(primitiveType &value) {
      add(primitiveTrie, value, keywordType::primitive);
    }

    void context_t::add(typedefType &value) {
      add(typedefTrie, value, keywordType::typedef_);
    }

    void context_t::add(classType &value) {
      add(classTrie, value, keywordType::class_);
    }

    void context_t::add(functionType &value) {
      add(functionTrie, value, keywordType::function_);
    }

    void context_t::add(attribute &value) {
      add(attributeTrie, value, keywordType::attribute);
    }

    void context_t::add(specifier &value,
                        const int ktype) {
      add(getTrie(ktype), value, ktype);
    }

    void context_t::add(statementTrie &trie,
                        specifier &value,
                        const int ktype) {
      // TODO: Add more information to the error message
      OCCA_ERROR("Keyword [" << (value.uniqueName()) << "] is already defined",
                 !keywordMap.has(value.uniqueName()));
      keywordMap.add(value.uniqueName(), keyword_t(ktype, &value));
      trie.add(value.uniqueName());
    }

    void context_t::addRecord(classType &value, statement_t &s) {
      addRecord(classTrie,
                &value,
                s,
                keywordType::class_);
    }

    void context_t::addRecord(functionType &value, statement_t &s) {
      addRecord(functionTrie,
                &value,
                s,
                keywordType::function_);
    }

    void context_t::addRecord(attribute &value, statement_t &s) {
      addRecord(attributeTrie,
                &value,
                s,
                keywordType::attribute);
    }

    void context_t::addRecord(statementTrie &trie,
                              specifier *ptr,
                              statement_t &s,
                              const int ktype) {
      statementTrie::result_t result = trie.get(ptr->uniqueName());
      OCCA_ERROR("Keyword [" << (ptr->uniqueName()) << "] is not in scope",
                 result.success());

      keyword_t keyword(ktype, ptr);
      statementKeywordMap &statements = result.value();
      statementKeywordMap::iterator it = statements.find(&s);
      if (it == statements.end()) {
        statements[&s].push_back(keyword);
        statementMap[&s].push_back(keyword);
      }
    }

    void context_t::removeRecord(statement_t &s) {
      statementKeywordMap::iterator it = statementMap.find(&s);
      if (it == statementMap.end()) {
        return;
      }
      keywordVector &keywords = it->second;
      const int keywordCount = (int) keywords.size();
      for (int i = 0; i < keywordCount; ++i) {
        statementKeywordMap &statements = getKeywordStatements(keywords[i]);
        statementKeywordMap::iterator it2 = statements.find(&s);
        if (it2 != statements.end()){
          statements.erase(it2);
        }
      }
      statementMap.erase(it);
    }

    statementPtrVector context_t::getStatements(classType &value) {
      statementPtrVector vec;
      getStatements(classTrie, value.uniqueName(), vec);
      return vec;
    }

    statementPtrVector context_t::getStatements(functionType &value) {
      statementPtrVector vec;
      getStatements(functionTrie, value.uniqueName(), vec);
      return vec;
    }

    statementPtrVector context_t::getStatements(attribute &value) {
      statementPtrVector vec;
      getStatements(attributeTrie, value.uniqueName(), vec);
      return vec;
    }

    void context_t::getStatements(statementTrie &trie,
                                  const std::string &name,
                                  statementPtrVector &vec) {
      statementKeywordMap &statements = (trie
                                         .get(name)
                                         .value());
      statementKeywordMap::iterator it = statements.begin();
      vec.clear();
      while (it != statements.end()) {
        vec.push_back(it->first);
        ++it;
      }
    }
  }
}
