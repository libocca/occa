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
    typedef void (preprocessor_t::*processDirective_t)(const char *&c);
    typedef trie_t<processDirective_t> directiveTrie_t;

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

    void processFile(const std::string &filename_);
    void process(const char *c);

    const macro_t* getMacro(const char *c, const size_t chars);

    std::string applyMacros(const char *c, const size_t chars);
    void applyMacros(const char *c, const size_t chars, std::string &out);

    void processDirective(const char *&c);
    void processIf(const char *&c);
    void processIfdef(const char *&c);
    void processIfndef(const char *&c);
    void processElif(const char *&c);
    void processElse(const char *&c);
    void processEndif(const char *&c);

    void processDefine(const char *&c);
    void processUndef(const char *&c);

    void processMessage(const char *&c, const bool isError);
    void processError(const char *&c);
    void processWarning(const char *&c);

    void processInclude(const char *&c);
    void processPragma(const char *&c);
    void processLine(const char *&c);

    //---[ Overriding Lex Methods ]-----
    void updateLines(const char *c, const int chars);

    void updatingSkipTo(const char *&c, const char delimiter);
    void updatingSkipTo(const char *&c, const char delimiter, const char escapeChar);
    void updatingSkipTo(const char *&c, const std::string &delimiters);
    void updatingSkipTo(const char *&c, const std::string &delimiters, const char escapeChar);

    void updatingSkipWhitespace(const char *&c);
    void updatingSkipToWhitespace(const char *&c);
    //==================================
  };
}
#endif
