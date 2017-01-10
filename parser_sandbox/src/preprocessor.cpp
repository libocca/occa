#include <sstream>
#include <stdlib.h>

#include "occa/tools/hash.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/lex.hpp"
#include "occa/tools/string.hpp"
#include "occa/parser/primitive.hpp"

#include "preprocessor.hpp"
#include "specialMacros.hpp"

namespace occa {
  //---[ Status ]---------------------
  static const int reading    = (1 << 0);
  static const int ignoring   = (1 << 1);
  static const int finishedIf = (1 << 2);

  preprocessor_t::status_t::status_t() {}

  preprocessor_t::status_t::status_t(const int status_, const int lineNumber_) :
    status(status_),
    lineNumber(lineNumber_) {}
  //==================================

  preprocessor_t& getPreprocessor(hash_t &compilerHash) {
    static std::map<hash_t, preprocessor_t> preprocessors;
    preprocessor_t &preprocessor = preprocessors[compilerHash];
    if (preprocessor.compilerMacros.isEmpty()) {
    }
    return preprocessor;
  }

  const std::string preprocessor_t::macroEndDelimiters = std::string(lex::whitespaceChars) + '(';

  preprocessor_t::preprocessor_t() :
    directives(getDirectiveTrie()) {

    filenameIdx = 0;
    filename    ="";
    lineNumber  = 0;

    compilerMacros.autoFreeze = false;
    macro_t *specialMacros[5] = {
      new fileMacro_t(this),   // __FILE__
      new lineMacro_t(this),   // __LINE__
      new dateMacro_t(this),   // __DATE__
      new timeMacro_t(this),   // __TIME__
      new counterMacro_t(this) // __COUNTER__
    };
    for (int i = 0; i < 5; ++i) {
      compilerMacros.add(specialMacros[i]->name, specialMacros[i]);
    }
    // [-] Add actual compiler macros as well

    outputStream = &std::cerr;

    pushStatus(reading);
  }

  void preprocessor_t::clear() {
    allFilenames.clear();
    filenameIdx = 0;

    filenames.clear();
    filename ="";

    filePointers.clear();

    lineNumbers.clear();
    lineNumber = 0;

    sourceMacros.clear();
    statusStack.clear();
  }

  void preprocessor_t::setOutputStream(std::ostream &outputStream_) {
    outputStream = &outputStream_;
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
    return statusStack[statusStack.size() - 1].status;
  }

  int& preprocessor_t::getStatusLineNumber() {
    return statusStack[statusStack.size() - 1].lineNumber;
  }

  void preprocessor_t::pushStatus(const int status) {
    statusStack.push_back(status_t(status, lineNumber));
  }

  void preprocessor_t::setFilename(const std::string &filename_, const bool add) {
    if (add) {
      allFilenames.add(filename_, 'u');
    }
    filename = filename_;
    filenameIdx = allFilenames.get(filename).valueIdx;
  }

  void preprocessor_t::processFile(const std::string &filename_) {
    setFilename(filename_);
    lineNumber = 0;

    char *c = io::c_read(filename);

    filenames.push_back(filename);
    filePointers.push_back(c);
    lineNumbers.push_back(lineNumber);

    process(c);
    ::free((void*) c);

    setFilename(filenames[filenames.size() - 1], false);
    lineNumber = lineNumbers[lineNumbers.size() - 1];

    filenames.pop_back();
    filePointers.pop_back();
    lineNumbers.pop_back();

    if ((filenames.size() == 0) &&
        (0 < statusStack.size())) {
      printError("#if without a closing #endif");
    }
  }

  void preprocessor_t::process(char *c) {
    while (*c != '\0') {
      updatingSkipWhitespace(c);
      if (*c == '#') {
        processDirective(++c);
      } else {
        char *cStart = c;
        updatingSkipTo(c, '\n');
        applyMacros(cStart, c - cStart);
      }
    }
  }

  const macro_t* preprocessor_t::getMacro(char *c, const size_t chars) {
    const std::string macroName = std::string(c, chars);

    macroTrie_t *macroSources[2] = { &sourceMacros, &compilerMacros };
    for (int i = 0; i < 2; ++i) {
      macroTrie_t::result_t result = macroSources[i]->get(macroName);
      if (result.success()) {
        macro_t * const macro = result.value();
        if ((macro->undefinedLine < 0) ||
            (lineNumber < macro->undefinedLine)) {
          return macro;
        }
      }
    }

    return NULL;
  }

  std::string preprocessor_t::applyMacros(const char *c) {
    std::string s(c);
    return applyMacros(&(s[0]), s.size());
  }

  std::string preprocessor_t::applyMacros(char *c, const size_t chars) {
    std::string out;
    applyMacros(c, chars, out);
    return out;
  }

  void preprocessor_t::applyMacros(char *c, const size_t chars, std::string &out) {
    char &lastChar = c[chars];
    const char lastCharValue = lastChar;
    lastChar = '\0';

    char *cStart = c;
    while (*c != '\0') {
      lex::skipWhitespace(c);
      char *cMacroStart = c;
      lex::skipTo(c, macroEndDelimiters);

      const macro_t *macro = getMacro(cMacroStart, c - cMacroStart);
      if (macro != NULL) {
        out += std::string(cStart, cMacroStart - cStart);
        out += macro->expand(c);
        cStart = c;
      }
    }

    if (cStart < c) {
      out += std::string(cStart, c - cStart);
    }
    lastChar = lastCharValue;
  }

  void preprocessor_t::processDirective(char *&c) {
    updatingSkipWhitespace(c);
    char *cStart = c;
    updatingSkipTo(c, macroEndDelimiters);
    char *cEnd = c;

    directiveTrie_t::result_t result = directives.get(cStart, cEnd - cStart);
    if (!result.success()) {
      std::string message = "Directive \"";
      message += std::string(cStart, cEnd - cStart);
      message += "\" is not defined";
      printError(message);
      updatingSkipTo(c, '\n');
      return;
    }

    // Parse #if[,def], #el[se,if], #endif even when ignored
    // For some reason the preprocessor honors ignored #if/#el/#endif stacks
    const int status = getStatus();
    if (!(status & ignoring) ||
        (strncmp(cStart, "if" , 2) &&
         strncmp(cStart, "el" , 2) &&
         strncmp(cStart, "end", 3))) {
      (this->*(result.value()))(c);
    } else {
      updatingSkipTo(c, '\n');
    }
  }

  void preprocessor_t::processIf(char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    int &status = getStatus();
    if (status & ignoring) {
      pushStatus(ignoring | finishedIf);
      return;
    }

    std::string line;
    applyMacros(cStart, c - cStart, line);

    status |= finishedIf;
  }

  void preprocessor_t::processIfdef(char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    int &status = getStatus();
    if (status & ignoring) {
      pushStatus(ignoring | finishedIf);
      return;
    }

    std::string line;
    applyMacros(cStart, c - cStart, line);
    const macro_t *macro = getMacro(&(line[0]), line.size());

    status |= (macro != NULL) ? ignoring : finishedIf;
  }

  void preprocessor_t::processIfndef(char *&c) {
    processIfdef(c);
    int &status = getStatus();
    // Ifdef already set finishedIf so we can return
    if (status & ignoring) {
      return;
    }
    // Do the opposite as Ifdef
    status ^= finishedIf;
  }

  void preprocessor_t::processElif(char *&c) {
    processIf(c);
  }

  void preprocessor_t::processElse(char *&c) {
    int &status = getStatus();
    if (status & finishedIf) {
      updatingSkipTo(c, '\n');
      return;
    }
    status |= finishedIf;
  }

  void preprocessor_t::processEndif(char *&c) {
    updatingSkipTo(c, '\n');

    if ((statusStack.size() == 0) ||
        (getStatus() & ignoring)) {
      printError("#endif without #if");
    }

    statusStack.pop_back();
  }

  void preprocessor_t::processDefine(char *&c) {
    const int thisLineNumber = lineNumber;
    char *cStart = c;
    updatingSkipTo(c, '\n');

    const char lastChar = *c;
    *c = '\0';
    macro_t *macro = new macro_t(this, cStart);
    *c = lastChar;

    macro->definedLine = thisLineNumber;
    sourceMacros.add(macro->name, macro);
  }

  void preprocessor_t::processUndef(char *&c) {
    const int thisLineNumber = lineNumber;
    char *cStart = c;
    updatingSkipToWhitespace(c);
    char *cEnd = c;
    updatingSkipTo(c, '\n');

    macroTrie_t::result_t result = sourceMacros.get(cStart, cEnd - cStart);
    if (0 <= result.valueIdx) {
      macro_t &macro = *(sourceMacros.values[result.valueIdx]);
      macro.undefinedLine = thisLineNumber;
    }
  }

  void preprocessor_t::processMessage(char *&c, const bool isError) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    std::string line;
    applyMacros(cStart, c - cStart, line);
    line = strip(line);

    if (isError) {
      printError(line);
    } else {
      printWarning(line);
    }
  }

  void preprocessor_t::processError(char *&c) {
    processMessage(c, true);
  }

  void preprocessor_t::processWarning(char *&c) {
    processMessage(c, false);
  }

  void preprocessor_t::processInclude(char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    std::string line;
    applyMacros(cStart, c - cStart, line);
    line = strip(line);

    processFile(io::filename(line));
  }

  void preprocessor_t::processPragma(char *&c) {
    updatingSkipTo(c, '\n');
  }

  void preprocessor_t::processLine(char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    std::string line;
    applyMacros(cStart, c - cStart, line);

    cStart = &(line[0]);
    char *cEnd = cStart + line.size();
    lex::strip(cStart, cEnd, '\\');

    // Get line number
    char *cStartLine = cStart;
    char *cEndLine = cStartLine;
    lex::skipToWhitespace(cEndLine);
    for (char *c_ = cStartLine; c_ < cEndLine; ++c_) {
      if (!lex::isDigit(*c_)) {
        printError("#line line number must be a simple number");
        return;
      }
    }
    const int cEndLineIdx = (int) (cEndLine - line.c_str());
    const int lineSize = (int) line.size();
    if (cEndLineIdx < lineSize) {
      line[cEndLineIdx] = '\0';
    }
    lineNumber = (int) primitive(cStartLine);

    if (lineSize <= cEndLineIdx) {
      return;
    }

    // Get filename (if exists)
    char *cStartFilename = cEndLine + 1;
    char *cEndFilename   = cEnd;
    lex::skipWhitespace(cStartFilename);
    if (cStartFilename < cEndFilename) {
      const std::string rawFilename = std::string(cStartFilename,
                                                  cEndFilename - cStartFilename);
      setFilename(io::filename(rawFilename));
    }
  }

  //---[ Messages ]---------------------
  void preprocessor_t::printMessage(const std::string &message,
                                    const int lineNumber_, const int position,
                                    const bool isError) const {
    std::stringstream ss;

    // /path/to/file:line:pos:
    const int filenameCount = (int) filenames.size();
    for (int i = 0; i < (filenameCount - 1); ++i) {
      ss << filenames[i] << ':' << lineNumbers[i] << '\n';
    }
    if (filenameCount == 0) {
      ss << "(source)" << ':' << lineNumber_;
      if (0 <= position) {
        ss << ':' << position;
      }
    } else {
      ss << filename << ':' << lineNumber_;
      if (0 <= position) {
        ss << ':' << position;
      }
    }

    // Error/Warning: <message>\n
    ss << ": "
       << (isError ? red("Error:") : yellow("Warning:"))
       << ' ' << message << '\n';

    // <line>
    if (filePointers.size()) {
      const char *c = filePointers[filePointers.size() - 1];
      int ln = 0;
      while ((*c != '\0') &&
             (ln != lineNumber_)) {
        lex::skipTo(c, '\n');
        ++c;
        ++ln;
      }

      if (ln == lineNumber_) {
        const char *cStart = c;
        lex::skipTo(c, '\n');
        ss << std::string(cStart, c - cStart);
      }

      // ... ^
      if (0 <= position) {
        ss << std::string(position, ' ') << green("^") << '\n';
      }
    }

    *outputStream << ss.str();
  }

  void preprocessor_t::printError(const std::string &message,
                                  const int lineNumber_, const int position) const {
    printMessage(message,
                 0 <= lineNumber_ ? lineNumber_ : lineNumber,
                 position,
                 true);
  }

  void preprocessor_t::printFatalError(const std::string &message,
                                       const int lineNumber_, const int position) const {
    printError(message, lineNumber_, position);
    ::exit(1);
  }

  void preprocessor_t::printWarning(const std::string &message,
                                    const int lineNumber_, const int position) const {
    printMessage(message,
                 0 <= lineNumber_ ? lineNumber_ : lineNumber,
                 position,
                 false);
  }
  //====================================

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
    lex::quotedSkipTo(c, delimiter);
    updateLines(cStart, c - cStart);
  }

  void preprocessor_t::updatingSkipTo(const char *&c, const std::string &delimiters) {
    const char *cStart = c;
    lex::quotedSkipTo(c, delimiters);
    updateLines(cStart, c - cStart);
  }

  void preprocessor_t::updatingSkipWhitespace(const char *&c) {
    const char *cStart = c;
    lex::skipWhitespace(c, '\\');
    updateLines(cStart, c - cStart);
  }

  void preprocessor_t::updatingSkipToWhitespace(const char *&c) {
    const char *cStart = c;
    lex::quotedSkipTo(c, lex::whitespaceChars);
    updateLines(cStart, c - cStart);
  }
  //====================================
}
