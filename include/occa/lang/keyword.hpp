#ifndef OCCA_INTERNAL_LANG_KEYWORD_HEADER
#define OCCA_INTERNAL_LANG_KEYWORD_HEADER

#include <map>
#include <string>

#include <occa/defines.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace lang {
    class keyword_t;
    class qualifier_t;
    class statementContext_t;
    class type_t;
    class token_t;
    class variable_t;
    class function_t;

    typedef std::map<std::string, keyword_t*> keywordMap;
    typedef keywordMap::iterator              keywordMapIterator;
    typedef keywordMap::const_iterator        cKeywordMapIterator;

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

    //---[ Keywords ]-------------------
    class keywords_t {
    public:
      keywordMap keywords;

      keywords_t();

      void free(const bool deleteSource = false);

      keywordMapIterator begin();
      keywordMapIterator end();

      template <class keywordType>
      void add(keywordType &keyword) {
        keywords[keyword.name()] = &keyword;
      }

      keyword_t& get(statementContext_t &smntContext,
                     token_t *token) const;
      keyword_t& get(statementContext_t &smntContext,
                     const std::string &name) const;
    };

    void freeKeywords(keywordMap &keywords,
                      const bool deleteSource = false);
    //==================================

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

    void getKeywords(keywords_t &keywords);
  }
}
#endif
