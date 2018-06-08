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

#include <map>
#include <string>

#include <occa/defines.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  namespace lang {
    class keyword_t;
    class qualifier_t;
    class type_t;
    class variable_t;
    class function_t;

    typedef std::map<std::string, keyword_t*> keywordMap;
    typedef keywordMap::iterator              keywordMapIterator;

    namespace keywordType {
      extern const int none;

      extern const int qualifier;
      extern const int type;
      extern const int variable;
      extern const int function;

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

      extern const int namespace_;

      extern const int public_;
      extern const int protected_;
      extern const int private_;
      extern const int classAccess;

      extern const int statement;
    }

    class keyword_t {
    public:
      virtual ~keyword_t();

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast keyword_t::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast keyword_t::to",
                   ptr != NULL);
        return *ptr;
      }

      virtual int type() const;
      virtual const std::string& name();

      virtual keyword_t* clone() const;

      virtual void deleteSource();

      virtual void printError(const std::string &message);

      static int safeType(keyword_t *keyword);
    };

    //---[ Qualifier ]------------------
    class qualifierKeyword : public keyword_t {
    public:
      const qualifier_t &qualifier;

      qualifierKeyword(const qualifier_t &qualifier_);

      virtual int type() const;
      virtual const std::string& name();

      virtual keyword_t* clone() const;
    };
    //==================================

    //---[ Type ]-----------------------
    class typeKeyword : public keyword_t {
    public:
      type_t &type_;

      typeKeyword(type_t &type__);

      virtual int type() const;
      virtual const std::string& name();

      virtual keyword_t* clone() const;

      virtual void deleteSource();

      virtual void printError(const std::string &message);
    };
    //==================================

    //---[ Variable ]-------------------
    class variableKeyword : public keyword_t {
    public:
      variable_t &variable;

      variableKeyword(variable_t &variable_);

      virtual int type() const;
      virtual const std::string& name();

      virtual keyword_t* clone() const;

      virtual void deleteSource();

      virtual void printError(const std::string &message);
    };
    //==================================

    //---[ Function ]-------------------
    class functionKeyword : public keyword_t {
    public:
      function_t &function;

      functionKeyword(function_t &function_);

      virtual int type() const;
      virtual const std::string& name();

      virtual keyword_t* clone() const;

      virtual void deleteSource();

      virtual void printError(const std::string &message);
    };
    //==================================

    //---[ Statement ]------------------
    class statementKeyword : public keyword_t {
    public:
      int sType;
      const std::string sName;

      statementKeyword(const int sType_,
                       const std::string &sName_);

      virtual int type() const;
      virtual const std::string& name();

      virtual keyword_t* clone() const;
    };
    //==================================

    void getKeywords(keywordMap &keywords);
    void freeKeywords(keywordMap &keywords,
                      const bool deleteSource = false);

    template <class keywordType>
    void addKeyword(keywordMap &keywords,
                    keywordType *keyword) {
      keywords[keyword->name()] = keyword;
    }

    void replaceKeyword(keywordMap &keywords,
                        keyword_t *keyword,
                        const bool deleteSource = false);
  }
}
#endif
