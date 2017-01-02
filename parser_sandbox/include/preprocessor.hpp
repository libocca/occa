#ifndef OCCA_PARSER_PREPROCESSOR_HEADER2
#define OCCA_PARSER_PREPROCESSOR_HEADER2

#include <vector>
#include <map>

#include "occa/defines.hpp"
#include "occa/types.hpp"

#include "macro.hpp"
#include "trie.hpp"

namespace occa {
  typedef trie_t<macro_t> macroTrie_t;

  class preprocessor_t;
  preprocessor_t& getPreprocessor(hash_t &compilerHash);

  class preprocessor_t {
  public:
    typedef void (preprocessor_t::*processDirective_t)(char *&c);
    typedef trie_t<processDirective_t> directiveTrie_t;

    static const std::string macroEndDelimiters;

    //---[ Stack Information ]----------
    trie_t<char> allFilenames;
    int filenameIdx;

    std::vector<std::string> filenames;
    std::string filename;

    std::vector<int> lineNumbers;
    int lineNumber;
    //==================================

    directiveTrie_t &directives;

    macroTrie_t compilerMacros;
    macroTrie_t sourceMacros;

    std::vector<int> statusStack;

    preprocessor_t();
    void clear();

    static directiveTrie_t& getDirectiveTrie();
    int& getStatus();

    void setFilename(const std::string &filename_, const bool add = true);
    void processFile(const std::string &filename_);

    void process(char *c);
    inline void process(const char *c) {
      std::string s(c);
      process(&(s[0]));
    }

    const macro_t* getMacro(char *c, const size_t chars);

    std::string applyMacros(char *c, const size_t chars);
    void applyMacros(char *c, const size_t chars, std::string &out);

    void processDirective(char *&c);
    void processIf(char *&c);
    void processIfdef(char *&c);
    void processIfndef(char *&c);
    void processElif(char *&c);
    void processElse(char *&c);
    void processEndif(char *&c);

    void processDefine(char *&c);
    void processUndef(char *&c);

    void processMessage(char *&c, const bool isError);
    void processError(char *&c);
    void processWarning(char *&c);

    void processInclude(char *&c);
    void processPragma(char *&c);
    void processLine(char *&c);

    //---[ Overriding Lex Methods ]-----
    void updateLines(const char *c, const int chars);

    void updatingSkipTo(const char *&c, const char delimiter);
    void updatingSkipTo(const char *&c, const std::string &delimiters);

    inline void updatingSkipTo(char *&c, const char delimiter) {
      updatingSkipTo((const char *&) c, delimiter);
    }

    inline void updatingSkipTo(char *&c, const std::string &delimiters) {
      updatingSkipTo((const char *&) c, delimiters);
    }

    void updatingSkipWhitespace(const char *&c);
    void updatingSkipToWhitespace(const char *&c);

    inline void updatingSkipWhitespace(char *&c) {
      updatingSkipWhitespace((const char *&) c);
    }

    inline void updatingSkipToWhitespace(char *&c) {
      updatingSkipToWhitespace((const char *&) c);
    }
    //==================================
  };
}
#endif
