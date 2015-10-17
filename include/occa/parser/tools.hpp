#ifndef OCCA_PARSER_TOOLS_HEADER
#define OCCA_PARSER_TOOLS_HEADER

#include "occa/parser/defines.hpp"
#include "occa/tools.hpp"

#include <iomanip>

namespace occa {
  //---[ Helper Functions ]-----------------------
  bool stringsAreEqual(const char *cStart, const size_t chars,
                       const char *c2);

  bool charIsIn(const char c, const char *delimiters);
  bool charIsIn2(const char *c, const char *delimiters);
  bool charIsIn3(const char *c, const char *delimiters);

  char upChar(const char c);
  char downChar(const char c);

  std::string upString(const char *c, const int chars);
  std::string upString(const std::string &s);

  bool upStringCheck(const std::string &a,
                     const std::string &b);

  template <class TM>
  inline std::string toString(const TM &t){
    std::stringstream ss;
    ss << t;
    return ss.str();
  }

  template <>
  inline std::string toString<float>(const float &t){
    std::stringstream ss;

    ss << std::scientific << std::setprecision(8) << t << 'f';

    return ss.str();
  }

  template <>
  inline std::string toString<double>(const double &t){
    std::stringstream ss;

    ss << std::scientific << std::setprecision(16) << t;

    return ss.str();
  }

  inline char back(std::string &s){
    return s[s.size() - 1];
  }

  bool isWhitespace(const char c);
  void skipWhitespace(const char *&c);
  void skipToWhitespace(const char *&c);

  int charsBeforeNewline(const std::string &str);

  bool isAString(const char *c);
  bool isAnInt(const char *c);
  bool isAFloat(const char *c);
  bool isANumber(const char *c);

  bool isAString(const std::string &str);
  bool isAnInt(const std::string &str);
  bool isAFloat(const std::string &str);
  bool isANumber(const std::string &str);

  inline bool isADigit(const char c){
    return (('0' <= c) && (c <= '9'));
  }

  void skipInt(const char *&c);

  void skipNumber(const char *&c,
                  const int parsingLanguage_ = parserInfo::parsingC);
  void skipFortranNumber(const char *&c);

  void skipString(const char *&c,
                  const int parsingLanguage_ = parserInfo::parsingC);

  char isAWordDelimiter(const char *c,
                        const int parsingLanguage_ = parserInfo::parsingC);
  char isAFortranWordDelimiter(const char *c);

  int skipWord(const char *&c,
               const int parsingLanguage_ = parserInfo::parsingC);

  bool isAnUpdateOperator(const std::string &s,
                          const int parsingLanguage_ = parserInfo::parsingC);
  bool isAnAssOperator(const std::string &s,
                       const int parsingLanguage_ = parserInfo::parsingC); // hehe
  bool isAnInequalityOperator(const std::string &s,
                              const int parsingLanguage_ = parserInfo::parsingC);

  const char* readLine(const char *c,
                       const int parsingLanguage_ = parserInfo::parsingC);
  const char* readFortranLine(const char *c);

  std::string compressWhitespace(const std::string &str);

  std::string strip(const char *c, const size_t chars,
                    const int parsingLanguage_ = parserInfo::parsingC);
  void strip(std::string &str,
             const int parsingLanguage_ = parserInfo::parsingC);

  char* cReadFile(const std::string &filename);

  int stripComments(std::string &line,
                    const int parsingLanguage_ = parserInfo::parsingC);
  int stripFortranComments(std::string &line);

  bool charStartsSection(const char c);
  bool charEndsSection(const char c);

  bool startsSection(const std::string &str);
  bool endsSection(const std::string &str);

  char segmentPair(const char c);
  char segmentPair(const std::string &str);
  void skipPair(const char *&c);

  int countDelimiters(const char *c, const char delimiter);

  void skipTo(const char *&c, const char delimiter);
  void skipTo(const char *&c, std::string delimiters);
  void skipToWord(const char *&c, std::string word);

  std::string findFileInPath(const std::string &filename);

  template <class TM>
  inline void swapValues(TM &a, TM &b){
    TM tmp = a;
    a      = b;
    b      = tmp;
  }
  //==============================================

  std::string getBits(const info_t value);

  //---[ Flag Holder ]----------------------------
  class flags_t {
  public:
    strToStrMap_t flags;

    inline flags_t(){};
    inline ~flags_t(){};

    inline flags_t(const flags_t &f) :
      flags(f.flags) {}

    inline flags_t& operator = (const flags_t &f){
      flags = f.flags;
      return *this;
    }

    inline std::string& operator [] (const std::string &flag){
      return flags[flag];
    }

    bool has(const std::string &flag);
    bool hasSet(const std::string &flag, const std::string &value);
    bool hasEnabled(const std::string &flag, bool defaultValue = false);
  };
  //==============================================
}

#endif
