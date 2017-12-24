#ifndef OCCA_PARSER_CONTEXT_HEADER2
#define OCCA_PARSER_CONTEXT_HEADER2

#include <map>
#include <vector>

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/sys.hpp"

#include "type.hpp"
#include "trie.hpp"
#include "keyword.hpp"

namespace occa {
  namespace lang {
    class statement;

    typedef std::vector<statement*> statementPtrVector_t;
    typedef std::vector<keyword_t>  keywordVector_t;

    typedef std::map<statement*, keywordVector_t> statementKeywordMap;
    typedef trie<statementKeywordMap>             statementTrie;
    typedef trie<keyword_t>                       keywordTrie;

    class context_t {
      statementKeywordMap statementMap;
      keywordTrie keywordMap;

      statementTrie qualifierTrie;
      statementTrie primitiveTrie;
      statementTrie typedefTrie;
      statementTrie classTrie;
      statementTrie functionTrie;
      statementTrie attributeTrie;

      statementTrie& getTrie(const int ktype);

      int getKeywordType(const std::string &name);

      statementKeywordMap& getKeywordStatements(keyword_t &keyword);

    public:
      void add(qualifier     &value);
      void add(primitiveType &value);
      void add(typedefType   &value);
      void add(classType     &value);
      void add(functionType  &value);
      void add(attribute     &value);
      void add(specifier &value,
               const int ktype);
      void add(statementTrie &trie,
               specifier &value,
               const int ktype);

      void addRecord(classType     &value, statement &s);
      void addRecord(functionType  &value, statement &s);
      void addRecord(attribute     &value, statement &s);
      void addRecord(statementTrie &trie,
                     specifier *ptr,
                     statement &s,
                     const int ktype);

      void removeRecord(statement &s);

      statementPtrVector_t getStatements(const std::string &name,
                                         const int ktype = keywordType::none);
      statementPtrVector_t getStatements(classType     &value);
      statementPtrVector_t getStatements(functionType  &value);
      statementPtrVector_t getStatements(attribute     &value);
      void getStatements(statementTrie &trie,
                         const std::string &name,
                         statementPtrVector_t &vec);
    };
  }
}

#endif
