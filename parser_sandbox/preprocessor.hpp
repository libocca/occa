#include "tools.hpp"
#include "macro.hpp"

typedef std::vector<macro_t>               macroVector_t;

typedef std::map<std::string, macroVector> macroVecMap_t;
typedef macroMap_t::iterator               macroVecMapIterator;
typedef macroMap_t::const_iterator         cMacroVecMapIterator;

typedef std::map<std::string, macro_t>     macroMap_t;
typedef macroMap_t::iterator               macroMapIterator;
typedef macroMap_t::const_iterator         cMacroMapIterator;

class preprocessor_t {
public:
  enum status_t {
    reading             = (1 << 0),

    ignoring            = (7 << 1),
    ignoringIf          = (1 << 1),
    ignoringElse        = (1 << 2),
    ignoringAll         = (1 << 3)
  };

  hash_t compilerHash;
  std::string language;

  macroMap_t compilerMacros;
  macroMap_t languageMacros;
  macroVecMap_t macros;

  std::vector<stringChunk> content;
  const char *contentMark;

  std::stack<status_t> statusStack;

  void clear() {
    macros.clear();
    content.clear();
    statusStack.clear();
  }

  status_t& getStatus() {
    return statusStack.top();
  }

  void addContent(const char *c) {
    content.push_back(stringChunk(contentMark, c - contentMark));
    contentMark = c;
  }

  void process(const char *c, std::string &output) {
    contentMark = c;
    while (*c != '\0') {
      occa::skipWhitespace(c);
      if (*c == '#') {
        loadMacro(++c);
      } else {
        const char *cStart = c;
        skipTo(c, '\n');
        applyMacros(cStart, c - cStart);
      }
    }
  }

  macro_t* getMacro(const char *c, const size_t chars, int lineNumber) {
    const std::string macroName = std::string(c, chars);
    macroVecMapIterator it = macros.find(macroName);
    if (it != macros.end()) {
      macroVector_t &foundMacros = it->second;
      const int foundMacroCount = (int) foundMacros.size();
      for (int i = 0; i < foundMacroCount; ++i) {
        macro_t &foundMacro = foundMacros[i];
        if ((foundMacro.definedLine <= lineNumber) &&
            ((foundMacro.undefinedLine < 0) ||
             (lineNumber <= foundMacro.undefinedLine))) {
          return &foundMacro;
        }
      }
    }
    macroMapIterator it2 = languageMacros.find(macroName);
    if (it2 != languageMacros.end()) {
      return &(it2->second);
    }
    it2 = compilerMacros.find(macroName);
    if (it2 != compilerMacros.end()) {
      return &(it2->second);
    }
    return NULL;
  }

  void applyMacros(const char *c, const size_t chars) {
    for (size_t i = 0; i < chars; ++i) {
      occa::skipWhitespace(c);
      const char *cStart = c;
      skipToWhitespace(c);

      macro_t *macro = getMacro(cStart, c - cStart);
      if (macro != NULL) {
      }
    }
  }

  void loadMacro(const char *&c) {
    const char *cHash = c;
    occa::skipWhitespace(c);
    const char *cStart = c;
    occa::skipToWhitespace(c);
    const char *cEnd = c;
    skipTo(c, '\n');

    switch (cEnd - cStart) {
    case 2:
      if (stringsAreEqual(cStart, (cEnd - cStart), "if")) {
        addContent(cHash);
        processIf(c);
      }
      break;
    case 4:
      if (stringsAreEqual(cStart, (cEnd - cStart), "else")) {
        addContent(cHash);
        processElse(c);
      } else if (stringsAreEqual(cStart, (cEnd - cStart), "elif")) {
        addContent(cHash);
        processElif(c);
      } else if (stringsAreEqual(cStart, (cEnd - cStart), "line")) {
        processLine(c);
      }
      break;
    case 5:
      if (stringsAreEqual(cStart, (cEnd - cStart), "error")) {
        processError(c);
      } else if (stringsAreEqual(cStart, (cEnd - cStart), "ifdef")) {
        addContent(cHash);
        processIfdef(c);
      } else if (stringsAreEqual(cStart, (cEnd - cStart), "endif")) {
        addContent(cHash);
        processEndif(c);
      } else if (stringsAreEqual(cStart, (cEnd - cStart), "undef")) {
        addContent(cHash);
        processUndef(c);
      }
      break;
    case 6:
      if (stringsAreEqual(cStart, (cEnd - cStart), "define")) {
        addContent(cHash);
        processDefine(c);
      } else if (stringsAreEqual(cStart, (cEnd - cStart), "ifndef")) {
        addContent(cHash);
        processIfndef(c);
      } else if (stringsAreEqual(cStart, (cEnd - cStart), "pragma")) {
        processPragma(c);
      }
      break;
    case 7:
      if (stringsAreEqual(cStart, (cEnd - cStart), "include")) {
        addContent(cHash);
        processInclude(c);
      } else if (stringsAreEqual(cStart, (cEnd - cStart), "warning")) {
        processWarning(c);
      }
      break;
    }
  }

  void processIf(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      statusStack.push(ignoringAll);
      return;
    }
  }

  void processElse(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      statusStack.push(ignoringAll);
      return;
    }
  }

  void processElif(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      statusStack.push(ignoringAll);
      return;
    }
  }

  void processLine(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }

  void processError(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }

  void processIfdef(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }

  void processEndif(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }

  void processUndef(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }

  void processDefine(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }

  void processIfndef(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }

  void processPragma(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }

  void processInclude(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }

  void processWarning(const char *&c) {
    status_t status = getStatus();
    if (status & ignoring) {
      return;
    }
  }
};