#ifndef OCCA_PARSER_TOOLS_HEADER
#define OCCA_PARSER_TOOLS_HEADER

#include "occaParserDefines.hpp"
#include "occaTools.hpp"

namespace occa {
  //---[ Helper Functions ]-----------------------
  std::string obfuscate(const std::string s1);
  std::string obfuscate(const std::string s1, const std::string s2);

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

  inline char back(std::string &s){
    return s[s.size() - 1];
  }

  bool isWhitespace(const char c);
  void skipWhitespace(const char *&c);
  void skipToWhitespace(const char *&c);

  bool isAString(const char *c);
  bool isAnInt(const char *c);
  bool isAFloat(const char *c);
  bool isANumber(const char *c);

  inline bool isANumber(const char c){
    return (('0' <= c) && (c <= '9'));
  }

  void skipInt(const char *&c);

  void skipNumber(const char *&c, const bool parsingC = true);
  void skipFortranNumber(const char *&c);

  void skipString(const char *&c, const bool parsingC = true);

  char isAWordDelimiter(const char *c, const bool parsingC = true);
  char isAFortranWordDelimiter(const char *c);

  int skipWord(const char *&c, const bool parsingC = true);

  bool isAnUpdateOperator(const std::string &s, const bool parsingC = true);
  bool isAnAssOperator(const std::string &s, const bool parsingC = true); // hehe
  bool isAnInequalityOperator(const std::string &s, const bool parsingC = true);

  const char* readLine(const char *c, const bool parsingC = true);
  const char* readFortranLine(const char *c);

  std::string compressWhitespace(const std::string &str);

  std::string strip(const char *c, const size_t chars, const bool parsingC = true);
  void strip(std::string &str, const bool parsingC = true);

  char* cReadFile(const std::string &filename);

  int stripComments(std::string &line, const bool parsingC = true);
  int stripFortranComments(std::string &line);

  char segmentPair(const char c);
  void skipPair(const char *&c);

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

  std::string getBits(const int value);
};

#endif
