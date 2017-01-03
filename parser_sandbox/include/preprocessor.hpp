#ifndef OCCA_PARSER_PREPROCESSOR_HEADER2
#define OCCA_PARSER_PREPROCESSOR_HEADER2

#include <ostream>
#include <vector>
#include <map>

#include "occa/defines.hpp"
#include "occa/types.hpp"

#include "macro.hpp"
#include "trie.hpp"

namespace occa {
  typedef trie_t<macro_t*> macroTrie_t;

  class preprocessor_t;
  preprocessor_t& getPreprocessor(hash_t &compilerHash);

  class preprocessor_t {
  public:
    typedef void (preprocessor_t::*processDirective_t)(char *&c);
    typedef trie_t<processDirective_t> directiveTrie_t;

    static const std::string macroEndDelimiters;

    class status_t {
    public:
      int status, lineNumber;

      status_t();
      status_t(const int status_, const int lineNumber_);
    };

    //---[ Stack Information ]----------
    trie_t<char> allFilenames;
    int filenameIdx;

    std::vector<std::string> filenames;
    std::string filename;

    std::vector<char*> filePointers;

    std::vector<int> lineNumbers;
    int lineNumber;

    std::vector<status_t> statusStack;
    //==================================

    //---[ Macros and Directives ]------
    directiveTrie_t &directives;

    macroTrie_t compilerMacros;
    macroTrie_t sourceMacros;
    //==================================

    //---[ Misc ]-----------------------
    std::ostream *outputStream;
    //==================================

    preprocessor_t();
    void clear();

    void setOutputStream(std::ostream &out_);

    static directiveTrie_t& getDirectiveTrie();

    int& getStatus();
    int& getStatusLineNumber();
    void pushStatus(const int status);

    void setFilename(const std::string &filename_, const bool add = true);
    void processFile(const std::string &filename_);

    void process(char *c);
    inline void process(const char *c) {
      std::string s(c);
      process(&(s[0]));
    }

    const macro_t* getMacro(char *c, const size_t chars);

    std::string applyMacros(const char *c);
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

    //---[ Messages ]-------------------
    void printMessage(const std::string &message,
                      const int lineNumber_, const int position,
                      const bool isError) const;

    void printError(const std::string &message,
                    const int lineNumber_ = -1, const int position = -1) const;

    void printFatalError(const std::string &message,
                         const int lineNumber_ = -1, const int position = -1) const;

    void printWarning(const std::string &message,
                      const int lineNumber_ = -1, const int position = -1) const;
    //==================================

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
