/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <algorithm>

#include "context.hpp"

namespace occa {
  namespace lang {
    statementTrie& context::getTrie(const int ktype) {
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

    int context::getKeywordType(const std::string &name) {
      keywordTrie::result_t result = keywordMap.get(name);
      return (result.success()
              ? result.value().ktype
              : keywordType::none);
    }

    statementKeywordMap& context::getKeywordStatements(keyword_t &keyword) {
      return (getTrie(keyword.ktype)
              .get(keyword.ptr->uniqueName())
              .value());
    }

    void context::add(qualifier &value) {
      add(qualifierTrie, value, keywordType::qualifier);
    }

    void context::add(primitiveType &value) {
      add(primitiveTrie, value, keywordType::primitive);
    }

    void context::add(typedefType &value) {
      add(typedefTrie, value, keywordType::typedef_);
    }

    void context::add(classType &value) {
      add(classTrie, value, keywordType::class_);
    }

    void context::add(functionType &value) {
      add(functionTrie, value, keywordType::function_);
    }

    void context::add(attribute_t &value) {
      add(attributeTrie, value, keywordType::attribute);
    }

    void context::add(specifier &value,
                      const int ktype) {
      add(getTrie(ktype), value, ktype);
    }

    void context::add(statementTrie &trie,
                      specifier &value,
                      const int ktype) {
      // TODO: Add more information to the error message
      OCCA_ERROR("Keyword [" << (value.uniqueName()) << "] is already defined",
                 !keywordMap.has(value.uniqueName()));
      keywordMap.add(value.uniqueName(), keyword_t(ktype, &value));
      trie.add(value.uniqueName());
    }

    void context::addRecord(classType &value, statement_t &s) {
      addRecord(classTrie,
                &value,
                s,
                keywordType::class_);
    }

    void context::addRecord(functionType &value, statement_t &s) {
      addRecord(functionTrie,
                &value,
                s,
                keywordType::function_);
    }

    void context::addRecord(attribute_t &value, statement_t &s) {
      addRecord(attributeTrie,
                &value,
                s,
                keywordType::attribute);
    }

    void context::addRecord(statementTrie &trie,
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

    void context::removeRecord(statement_t &s) {
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

    statementPtrVector context::getStatements(classType &value) {
      statementPtrVector vec;
      getStatements(classTrie, value.uniqueName(), vec);
      return vec;
    }

    statementPtrVector context::getStatements(functionType &value) {
      statementPtrVector vec;
      getStatements(functionTrie, value.uniqueName(), vec);
      return vec;
    }

    statementPtrVector context::getStatements(attribute_t &value) {
      statementPtrVector vec;
      getStatements(attributeTrie, value.uniqueName(), vec);
      return vec;
    }

    void context::getStatements(statementTrie &trie,
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
