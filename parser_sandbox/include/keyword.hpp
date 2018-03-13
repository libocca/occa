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
#ifndef OCCA_LANG_KEYWORD_HEADER
#define OCCA_LANG_KEYWORD_HEADER

#include "trie.hpp"

namespace occa {
  namespace lang {
    class keyword_t;
    class qualifier_t;
    class type_t;
    class variable;

    typedef trie<keyword_t*> keywordTrie;

    namespace keywordType {
      extern const int none;

      extern const int qualifier;
      extern const int type;
      extern const int variable;

      extern const int if_;
      extern const int else_;
      extern const int switch_;
      extern const int conditional;

      extern const int case_;
      extern const int default_;
      extern const int switchLabel;

      extern const int for_;
      extern const int while_;
      extern const int do_;
      extern const int iteration;

      extern const int break_;
      extern const int continue_;
      extern const int return_;
      extern const int goto_;
      extern const int jump;

      extern const int statement;
    }

    class keyword_t {
    public:
      virtual ~keyword_t();

      virtual int type() = 0;
      virtual std::string name() = 0;
    };

    //---[ Qualifier ]------------------
    class qualifierKeyword : public keyword_t {
    public:
      const qualifier_t &qualifier;

      qualifierKeyword(const qualifier_t &qualifier_);

      virtual int type();
      virtual std::string name();
    };
    //==================================

    //---[ Type ]-----------------------
    class typeKeyword : public keyword_t {
    private:
      const type_t &type_;

    public:
      typeKeyword(const type_t &type__);

      virtual int type();
      virtual std::string name();
    };
    //==================================

    //---[ Variable ]-------------------
    class variableKeyword : public keyword_t {
    private:
      const variable &var;

    public:
      variableKeyword(const variable &var_);

      virtual int type();
      virtual std::string name();
    };
    //==================================

    //---[ Statement ]------------------
    class statementKeyword : public keyword_t {
    public:
      int sType;
      const std::string sName;

      statementKeyword(const int sType_,
                       const std::string &sName_);

      virtual int type();
      virtual std::string name();
    };
    //==================================

    void getKeywords(keywordTrie &keywords);
    void freeKeywords(keywordTrie &keywords);

    template <class keywordType>
    void addKeyword(keywordTrie &keywords,
                    keywordType *keyword) {
      keywords.add(keyword->name(),
                   keyword);
    }
  }
}
#endif
