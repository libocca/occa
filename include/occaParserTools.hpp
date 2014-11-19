#ifndef OCCA_PARSER_TOOLS_HEADER
#define OCCA_PARSER_TOOLS_HEADER

#include "occaParserDefines.hpp"

namespace occa {
  namespace parserNamespace {
    //---[ Helper Functions ]-----------------------
    std::string obfuscate(const std::string s1);
    std::string obfuscate(const std::string s1, const std::string s2);

    bool stringsAreEqual(const char *cStart, const size_t chars,
                         const char *c2);

    bool charIsIn(const char c, const char *delimeters);
    bool charIsIn2(const char *c, const char *delimeters);
    bool charIsIn3(const char *c, const char *delimeters);

    bool isWhitespace(const char c);
    void skipWhitespace(const char *&c);
    void skipToWhitespace(const char *&c);

    bool isAString(const char *c);
    bool isAnInt(const char *c);
    bool isAFloat(const char *c);
    bool isANumber(const char *c);

    void skipInt(const char *&c);
    void skipNumber(const char *&c);
    void skipString(const char *&c);

    char isAWordDelimeter(const char *c);

    int skipWord(const char *&c);

    const char* readLine(const char *c);

    std::string compressWhitespace(const std::string &str);

    std::string strip(const char *c, const size_t chars);
    void strip(std::string &str);

    char* cReadFile(const std::string &filename);

    int stripComments(std::string &line);

    char segmentPair(const char c);
    void skipPair(const char *&c);

    template <class TM>
    inline void swapValues(TM &a, TM &b){
      TM tmp = a;
      a      = b;
      b      = tmp;
    }
    //==============================================

    std::string getBits(const int value);
  };
};

#endif
