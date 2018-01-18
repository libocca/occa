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
    class statement_t;

    typedef std::vector<statement_t*>           statementPtrVector;
    typedef std::vector<keyword_t>              keywordVector;

    typedef std::map<statement_t*, keywordVector> statementKeywordMap;
    typedef trie<statementKeywordMap>             statementTrie;
    typedef trie<keyword_t>                       keywordTrie;

    class context {
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
      void add(attribute_t   &value);
      void add(specifier &value,
               const int ktype);
      void add(statementTrie &trie,
               specifier &value,
               const int ktype);

      void addRecord(classType     &value, statement_t &s);
      void addRecord(functionType  &value, statement_t &s);
      void addRecord(attribute_t   &value, statement_t &s);
      void addRecord(statementTrie &trie,
                     specifier *ptr,
                     statement_t &s,
                     const int ktype);

      void removeRecord(statement_t &s);

      statementPtrVector getStatements(const std::string &name,
                                       const int ktype = keywordType::none);
      statementPtrVector getStatements(classType     &value);
      statementPtrVector getStatements(functionType  &value);
      statementPtrVector getStatements(attribute_t   &value);
      void getStatements(statementTrie &trie,
                         const std::string &name,
                         statementPtrVector &vec);
    };
  }
}

#endif
