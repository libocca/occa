#include "occa/tools/hash.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/lex.hpp"

#include "preprocessor.hpp"

namespace occa {
  //---[ Status ]---------------------
  static const int reading    = (1 << 0);
  static const int ignoring   = (1 << 1);
  static const int finishedIf = (1 << 2);
  //==================================

  preprocessor_t& getPreprocessor(hash_t &compilerHash) {
    static std::map<hash_t, preprocessor_t> preprocessors;
    preprocessor_t &preprocessor = preprocessors[compilerHash];
    if (preprocessor.compilerMacros.isEmpty()) {
    }
    return preprocessor;
  }

  preprocessor_t::preprocessor_t() :
    directives(getDirectiveTrie()) {
    filenameIdx = 0;
    filename    ="";
    lineNumber  = 0;

    compilerMacros.autoFreeze = false;

    statusStack.push_back(reading);
  }

  void preprocessor_t::clear() {
    allFilenames.clear();
    filenameIdx = 0;

    filenames.clear();
    filename ="";

    lineNumbers.clear();
    lineNumber = 0;

    sourceMacros.clear();
    statusStack.clear();
  }

  preprocessor_t::directiveTrie_t& preprocessor_t::getDirectiveTrie() {
    static directiveTrie_t trie;
    if (trie.isEmpty()) {
      trie.autoFreeze = false;
      trie.add("if"    , &preprocessor_t::processIf);
      trie.add("ifdef" , &preprocessor_t::processIfdef);
      trie.add("ifndef", &preprocessor_t::processIfndef);
      trie.add("elif"  , &preprocessor_t::processElif);
      trie.add("else"  , &preprocessor_t::processElse);
      trie.add("endif" , &preprocessor_t::processEndif);

      trie.add("define", &preprocessor_t::processDefine);
      trie.add("undef" , &preprocessor_t::processUndef);

      trie.add("error"  , &preprocessor_t::processError);
      trie.add("warning", &preprocessor_t::processWarning);

      trie.add("include", &preprocessor_t::processInclude);
      trie.add("pragma" , &preprocessor_t::processPragma);
      trie.add("line"   , &preprocessor_t::processLine);
      trie.freeze();
    }
    return trie;
  }

  int& preprocessor_t::getStatus() {
    return statusStack[statusStack.size() - 1];
  }

  void preprocessor_t::processFile(const std::string &filename_) {
    allFilenames.add(filename_, 'u');
    filenames.push_back(filename);
    lineNumbers.push_back(lineNumber);

    filename     = filename_;
    lineNumber   = 0;
    filenameIdx  = allFilenames.get(filename).valueIdx;

    const char *c = io::c_read(filename);
    process(c);
    ::free((void*) c);

    filename    = filenames[filenames.size() - 1];
    lineNumber  = lineNumbers[lineNumbers.size() - 1];
    filenameIdx = allFilenames.get(filename).valueIdx;

    filenames.pop_back();
    lineNumbers.pop_back();
  }

  void preprocessor_t::process(const char *c) {
    while (*c != '\0') {
      updatingSkipWhitespace(c);
      if (*c == '#') {
        processDirective(++c);
      } else {
        const char *cStart = c;
        updatingSkipTo(c, '\n');
        applyMacros(cStart, c - cStart);
      }
    }
  }

  const macro_t* preprocessor_t::getMacro(const char *c, const size_t chars) {
    const std::string macroName = std::string(c, chars);
    const int macroNameLength = (int) macroName.size();

    macroTrie_t *macroSources[2] = {&sourceMacros, &compilerMacros};
    for (int i = 0; i < 2; ++i) {
      macroTrie_t::result_t result = macroSources[i]->get(macroName);
      if (result.length == macroNameLength) {
        const macro_t &macro = result.value();
        if ((macro.undefinedLine < 0) ||
            (lineNumber < macro.undefinedLine)) {
          return &macro;
        }
      }
    }

    return NULL;
  }

  std::string preprocessor_t::applyMacros(const char *c, const size_t chars) {
    std::string out;
    applyMacros(c, chars, out);
    return out;
  }

  void preprocessor_t::applyMacros(const char *c, const size_t chars, std::string &out) {
    for (size_t i = 0; i < chars; ++i) {
      updatingSkipWhitespace(c);
      const char *cStart = c;
      updatingSkipToWhitespace(c);

      const macro_t *macro = getMacro(cStart, c - cStart);
      if (macro == NULL) {
        return;
      }
    }
  }

  void preprocessor_t::processDirective(const char *&c) {
    const int thisLineNumber = lineNumber;
    static std::string delimiters;
    if (delimiters.size() == 0) {
      delimiters = lex::whitespaceChars;
      delimiters += '(';
    }
    updatingSkipWhitespace(c);
    const char *cStart = c;
    updatingSkipTo(c, delimiters);
    const char *cEnd = c;

    directiveTrie_t::result_t result = directives.get(cStart, cEnd - cStart);
    OCCA_ERROR("Directive \""
               << std::string(cStart, cEnd - cStart)
               << "\" is not defined",
               0 <= result.valueIdx);

    // Parse #if[,def], #el[se,if], #endif even when ignored
    // For some reason the preprocessor honors ignored #if/#el/#endif stacks
    const int status = getStatus();
    if (!(status & ignoring) ||
        (strncmp(cStart, "if" , 2) &&
         strncmp(cStart, "el" , 2) &&
         strncmp(cStart, "end", 3))) {
      (this->*(result.value()))(c);
    } else {
      updatingSkipTo(c, '\n', '\\');
    }
  }

  void preprocessor_t::processIf(const char *&c) {
    const char *cStart = c;
    updatingSkipTo(c, '\n', '\\');

    int &status = getStatus();
    if (status & ignoring) {
      statusStack.push_back(ignoring | finishedIf);
      return;
    }

    std::string line;
    applyMacros(cStart, c - cStart, line);

    status |= finishedIf;
  }

  void preprocessor_t::processIfdef(const char *&c) {
    const char *cStart = c;
    updatingSkipTo(c, '\n', '\\');

    int &status = getStatus();
    if (status & ignoring) {
      statusStack.push_back(ignoring | finishedIf);
      return;
    }

    std::string line;
    applyMacros(cStart, c - cStart, line);
    const macro_t *macro = getMacro(line.c_str(), line.size());

    status |= (macro != NULL) ? ignoring : finishedIf;
  }

  void preprocessor_t::processIfndef(const char *&c) {
    processIfdef(c);
    int &status = getStatus();
    // Ifdef already set finishedIf so we can return
    if (status & ignoring) {
      return;
    }
    // Do the opposite as Ifdef
    status ^= finishedIf;
  }

  void preprocessor_t::processElif(const char *&c) {
    processIf(c);
  }

  void preprocessor_t::processElse(const char *&c) {
    int &status = getStatus();
    if (status & finishedIf) {
      updatingSkipTo(c, '\n', '\\');
      return;
    }
    status |= finishedIf;
  }

  void preprocessor_t::processEndif(const char *&c) {
    updatingSkipTo(c, '\n', '\\');
    if (getStatus() & ignoring) {
      statusStack.pop_back();
      return;
    }
  }

  void preprocessor_t::processDefine(const char *&c) {
    const int thisLineNumber = lineNumber;
    const char *cStart = c;
    updatingSkipTo(c, '\n', '\\');

    macro_t macro(cStart, c - cStart);
    macro.definedLine = thisLineNumber;
    sourceMacros.add(macro.name, macro);
  }

  void preprocessor_t::processUndef(const char *&c) {
    const int thisLineNumber = lineNumber;
    const char *cStart = c;
    updatingSkipToWhitespace(c);
    const char *cEnd = c;
    updatingSkipTo(c, '\n', '\\');

    macroTrie_t::result_t result = sourceMacros.get(cStart, cEnd - cStart);
    if (0 <= result.valueIdx) {
      macro_t &macro = sourceMacros.values[result.valueIdx];
      macro.undefinedLine = thisLineNumber;
    } else {
      OCCA_FORCE_WARNING("Macro not defined");
    }
  }

  void preprocessor_t::processMessage(const char *&c, const bool isError) {
    const int thisLineNumber = lineNumber;
    const char *cStart = c;
    updatingSkipWhitespace(c);
    OCCA_FORCE_ERROR("Incorrect #include");
  }

  void preprocessor_t::processError(const char *&c) {
    processMessage(c, true);
  }

  void preprocessor_t::processWarning(const char *&c) {
    processMessage(c, false);
  }

  void preprocessor_t::processInclude(const char *&c) {
  }

  void preprocessor_t::processPragma(const char *&c) {
  }

  void preprocessor_t::processLine(const char *&c) {
  }

  //---[ Overriding Lex Methods ]-------
  void preprocessor_t::updateLines(const char *c, const int chars) {
    for (int i = 0; i < chars; ++i) {
      if (c[i] == '\n') {
        ++lineNumber;
      }
    }
  }

  void preprocessor_t::updatingSkipTo(const char *&c, const char delimiter) {
    const char *cStart = c;
    lex::skipTo(c, delimiter);
    updateLines(cStart, c - cStart);
  }

  void preprocessor_t::updatingSkipTo(const char *&c, const char delimiter, const char escapeChar) {
    const char *cStart = c;
    lex::skipTo(c, delimiter, escapeChar);
    updateLines(cStart, c - cStart);
  }

  void preprocessor_t::updatingSkipTo(const char *&c, const std::string &delimiters) {
    const char *cStart = c;
    lex::skipTo(c, delimiters);
    updateLines(cStart, c - cStart);
  }

  void preprocessor_t::updatingSkipTo(const char *&c, const std::string &delimiters, const char escapeChar) {
    const char *cStart = c;
    lex::skipTo(c, delimiters, escapeChar);
    updateLines(cStart, c - cStart);
  }

  void preprocessor_t::updatingSkipWhitespace(const char *&c) {
    const char *cStart = c;
    lex::skipWhitespace(c, '\\');
    updateLines(cStart, c - cStart);
  }

  void preprocessor_t::updatingSkipToWhitespace(const char *&c) {
    const char *cStart = c;
    lex::skipToWhitespace(c);
    updateLines(cStart, c - cStart);
  }
  //====================================
}
