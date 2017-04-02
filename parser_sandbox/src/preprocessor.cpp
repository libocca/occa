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
  //---[ Stack Information ]------------
  //|-----[ Status ]--------------------
  static const int reading    = (1 << 0);
  static const int ignoring   = (1 << 1);
  static const int foundElse  = (1 << 2);
  static const int finishedIf = (1 << 3);
  //|===================================

  preprocessor_t::status_t::status_t() {}

  preprocessor_t::status_t::status_t(const int status_, const char *filePosition_) :
    status(status_),
    filePosition(filePosition_) {}

  void preprocessor_t::status_t::clear() {
    status = reading;
    filePosition = NULL;
  }

  preprocessor_t::frame_t::frame_t(const preprocessor_t *preprocessor_) :
    preprocessor(preprocessor_) {
    clear();
  }

  void preprocessor_t::frame_t::clear() {
    filenameIdx  = 0;
    fileStart    = NULL;
    fileEnd      = NULL;
    filePosition = NULL;
    lineNumber   = 1;
  }

  std::string preprocessor_t::frame_t::filename() const {
    return preprocessor->allFilenames.values[filenameIdx];
  }

  std::string preprocessor_t::frame_t::getLineMessage() const {
    return getLineMessage(lineNumber);
  }

  std::string preprocessor_t::frame_t::getLineMessage(const int lineNumber_) const {
    std::string ret;
    ret += filename();
    ret += ':';
    ret += occa::toString(lineNumber_);
    return ret;
  }
  //====================================

  preprocessor_t& getPreprocessor(hash_t &compilerHash) {
    static std::map<hash_t, preprocessor_t> preprocessors;
    preprocessor_t &preprocessor = preprocessors[compilerHash];
    if (preprocessor.compilerMacros.isEmpty()) {
    }
    return preprocessor;
  }

  const std::string preprocessor_t::macroEndDelimiters = std::string(lex::whitespaceChars) + '(';

  preprocessor_t::preprocessor_t() :
    currentFrame(this),
    directives(getDirectiveTrie()) {

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

    // Alternative representations
    compilerMacros.add("and"   , new macro_t(this, "and     &&"));
    compilerMacros.add("and_eq", new macro_t(this, "and_eq  &="));
    compilerMacros.add("bitand", new macro_t(this, "bitand  &"));
    compilerMacros.add("bitor" , new macro_t(this, "bitor   |"));
    compilerMacros.add("compl" , new macro_t(this, "compl   ~"));
    compilerMacros.add("not"   , new macro_t(this, "not     !"));
    compilerMacros.add("not_eq", new macro_t(this, "not_eq  !="));
    compilerMacros.add("or"    , new macro_t(this, "or      ||"));
    compilerMacros.add("or_eq" , new macro_t(this, "or_eq   |="));
    compilerMacros.add("xor"   , new macro_t(this, "xor     ^"));
    compilerMacros.add("xor_eq", new macro_t(this, "xor_eq  ^="));

    // [-] Add actual compiler macros as well

    exitOnFatalError = true;
    errorCount   = 0;
    warningCount = 0;

    outputStream = &std::cerr;

    pushStatus(reading);
  }

  void preprocessor_t::clear() {
    allFilenames.clear();

    frames.clear();
    currentFrame.clear();

    statusStack.clear();
    currentStatus.clear();

    sourceMacros.clear();
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

  void preprocessor_t::pushStatus(const int status) {
    statusStack.push_back(currentStatus);
    currentStatus.status       = status;
    currentStatus.filePosition = currentFrame.filePosition;
  }

  int preprocessor_t::popStatus() {
    const int pos = (int) statusStack.size() - 1;
    if (pos >= 0) {
      currentStatus.status       = statusStack[pos].status;
      currentStatus.filePosition = statusStack[pos].filePosition;
      statusStack.pop_back();
      return currentStatus.status;
    }
    return 0;
  }

  void preprocessor_t::setFilename(const std::string &filename, const bool add) {
    if (add) {
      allFilenames.add(filename, filename);
    }
    currentFrame.filenameIdx = allFilenames.get(filename).valueIdx;
  }

  void preprocessor_t::processFile(const std::string &filename) {
    char *c = io::c_read(filename);
    processFile(filename, c);
    ::free((void*) c);
  }

  void preprocessor_t::processFile(const std::string &filename, char *content) {
    if (currentFrame.fileStart) {
      frames.push_back(currentFrame);
    }
    currentFrame.clear();
    setFilename(filename);
    currentFrame.fileStart    = content;
    currentFrame.fileEnd      = content + strlen(content);
    currentFrame.filePosition = content;

    process(currentFrame.filePosition);

    if (frames.size()) {
      currentFrame = frames[frames.size() - 1];
      frames.pop_back();

      if (!frames.size() && statusStack.size()) {
        printError("#if without a closing #endif",
                   currentStatus.filePosition);
      }
    }
  }

  void preprocessor_t::processSource(char *c) {
    processFile("(source)", c);
  }

  void preprocessor_t::processSource(const char *c) {
    std::string s(c);
    processFile("(source)", &(s[0]));
  }

  void preprocessor_t::process(char *&c) {
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
            (currentFrame.lineNumber < macro->undefinedLine)) {
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
      printError(message, cStart);
      updatingSkipTo(c, '\n');
      return;
    }

    // Parse #if[,def], #el[se,if], #end[if] even when ignored
    // The preprocessor honors ignored #if/#el/#endif stacks
    const int status = currentStatus.status;

    if (!(status & ignoring) ||
        (strncmp(cStart, "if" , 2) &&
         strncmp(cStart, "el" , 2) &&
         strncmp(cStart, "end", 3))) {
      (this->*(result.value()))(cStart, c);
    } else {
      updatingSkipTo(c, '\n');
    }
  }

  void preprocessor_t::processIf(char *&dStart, char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    int status = currentStatus.status;

    if (!(status & ignoring)) {
      std::string line;
      applyMacros(cStart, c - cStart, line);
      pushStatus(eval<bool>(line) ? reading : ignoring);
    } else {
      pushStatus(finishedIf);
    }
  }

  void preprocessor_t::processIfdef(char *&dStart, char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    int status = currentStatus.status;

    if (!(status & ignoring)) {
      std::string line;
      applyMacros(cStart, c - cStart, line);

      const macro_t *macro = getMacro(&(line[0]), line.size());
      pushStatus(macro != NULL ? reading : ignoring);
    } else {
      pushStatus(finishedIf);
    }
  }

  void preprocessor_t::processIfndef(char *&dStart, char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    int status = currentStatus.status;

    if (!(status & ignoring)) {
      std::string line;
      applyMacros(cStart, c - cStart, line);

      const macro_t *macro = getMacro(&(line[0]), line.size());
      pushStatus(macro == NULL ? reading : ignoring);
    } else {
      pushStatus(finishedIf);
    }
  }

  void preprocessor_t::processElif(char *&dStart, char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    int status = currentStatus.status;

    if (!statusStack.size()) {
      printError("#elif without #if", cStart);
    }

    if (status & ignoring) {
      std::string line;
      applyMacros(cStart, c - cStart, line);
      pushStatus(eval<bool>(line) ? reading : ignoring);
    } else {
      pushStatus(finishedIf);
    }
  }

  void preprocessor_t::processElse(char *&dStart, char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    int status = currentStatus.status;

    if (!statusStack.size()) {
      printError("#else without #if", cStart);
    }
    if (status & foundElse) {
      printError("Found two #else directives", cStart);
    }

    if (status & ignoring) {
      std::string line;
      applyMacros(cStart, c - cStart, line);
      pushStatus(foundElse | (eval<bool>(line) ? reading : ignoring));
    } else {
      pushStatus(foundElse | finishedIf);
    }
  }

  void preprocessor_t::processEndif(char *&dStart, char *&c) {
    const char *cStart = c;
    updatingSkipTo(c, '\n');

    if (!popStatus()) {
      printError("#endif without #if", cStart);
    }
  }

  void preprocessor_t::processDefine(char *&dStart, char *&c) {
    const int thisLineNumber = currentFrame.lineNumber;
    char *cStart = c;
    updatingSkipTo(c, '\n');

    const char lastChar = *c;
    *c = '\0';
    macro_t *macro = new macro_t(this, cStart);
    *c = lastChar;

    macro->definedLine = thisLineNumber;
    sourceMacros.add(macro->name, macro);
  }

  void preprocessor_t::processUndef(char *&dStart, char *&c) {
    const int thisLineNumber = currentFrame.lineNumber;
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
      printError(line, cStart);
    } else {
      printWarning(line, cStart);
    }
  }

  void preprocessor_t::processError(char *&dStart, char *&c) {
    processMessage(dStart, true);
  }

  void preprocessor_t::processWarning(char *&dStart, char *&c) {
    processMessage(dStart, false);
  }

  void preprocessor_t::processInclude(char *&dStart, char *&c) {
    char *cStart = c;
    updatingSkipTo(c, '\n');

    std::string line;
    applyMacros(cStart, c - cStart, line);
    line = strip(line);

    processFile(io::filename(line));
  }

  void preprocessor_t::processPragma(char *&dStart, char *&c) {
    updatingSkipTo(c, '\n');
  }

  void preprocessor_t::processLine(char *&dStart, char *&c) {
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
        printError("#line line number must be a simple number", c_);
        return;
      }
    }
    const int cEndLineIdx = (int) (cEndLine - line.c_str());
    const int lineSize = (int) line.size();
    if (cEndLineIdx < lineSize) {
      line[cEndLineIdx] = '\0';
    }
    // We subtract one since the NEXT line has the line number given in #line
    currentFrame.lineNumber = ((int) primitive(cStartLine)) - 1;

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
                                    const char *errorPosition,
                                    const bool isError) const {
    std::stringstream ss;
    const char *fileStart = currentFrame.fileStart;
    const char *fileEnd   = currentFrame.fileEnd;

    const bool useErrorPosition = (fileStart                    &&
                                   (fileStart <= errorPosition) &&
                                   (errorPosition < fileEnd));

    // Print filename stack
    const int frameCount = frames.size();
    for (int i = 0; i < frameCount; ++i) {
      ss << frames[i].getLineMessage() << '\n';
    }

    // Get line number based on errorPosition
    int lineNumber = currentFrame.lineNumber;
    const char *lineStart = fileStart;

    if (useErrorPosition) {
      lineNumber = 1;
      for (const char *c = fileStart; c < errorPosition; ++c) {
        if (*c == '\n') {
          lineStart = c + 1;
          ++lineNumber;
        }
      }
    } else {
      for (int i = 0; i < lineNumber; ++i) {
        lex::skipTo(lineStart, '\n');
        lineStart += (lineStart != '\0');
      }
    }
    // Print current file with found line number
    ss << currentFrame.getLineMessage(lineNumber);

    // Error/Warning: <message>\n
    ss << ": "
       << (isError ? red("Error:") : yellow("Warning:"))
       << ' ' << message << '\n';

    // Print found line
    const char *lineEnd = lineStart;
    lex::skipTo(lineEnd, '\n');

    if (lineStart < lineEnd) {
      ss << std::string(lineStart, lineEnd - lineStart) << '\n';

      // ... ^
      if (useErrorPosition) {
        ss << std::string(errorPosition - lineStart, ' ') << green("^") << '\n';
      }
    }

    *outputStream << ss.str();
  }

  void preprocessor_t::printError(const std::string &message,
                                  const char *errorPosition) const {
    printMessage(message, errorPosition, true);
    ++errorCount;
  }

  void preprocessor_t::printFatalError(const std::string &message,
                                       const char *errorPosition) const {
    printMessage(message, errorPosition, true);
    ++errorCount;
    if (exitOnFatalError) {
      printErrorAndWarningCounts();
      throw 1;
    }
  }

  void preprocessor_t::printWarning(const std::string &message,
                                    const char *errorPosition) const {
    printMessage(message, errorPosition, false);
    ++warningCount;
  }

  void preprocessor_t::printErrorAndWarningCounts() const {
    if (warningCount) {
      *outputStream << warningCount << " warning";
      if (1 < warningCount) {
        *outputStream << 's';
      }
    }
    if (errorCount) {
      if (warningCount) {
        *outputStream << " and ";
      }
      *outputStream << errorCount << " error";
      if (1 < errorCount) {
        *outputStream << 's';
      }
    }
    if (warningCount || errorCount) {
      *outputStream << " generated.\n";
    }
  }
  //====================================

  //---[ Overriding Lex Methods ]-------
  void preprocessor_t::updateLines(const char *c, const int chars) {
    for (int i = 0; i < chars; ++i) {
      if (c[i] == '\n') {
        ++currentFrame.lineNumber;
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
