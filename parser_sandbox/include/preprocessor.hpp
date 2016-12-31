#ifndef OCCA_PARSER_PREPROCESSOR_HEADER2
#define OCCA_PARSER_PREPROCESSOR_HEADER2

#include <vector>
#include <map>

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/hash.hpp"
#include "occa/tools/lex.hpp"

#include "macro.hpp"

#  if 0
namespace occa {
  typedef std::vector<macro_t>                 macroVector_t;

  typedef std::map<std::string, macroVector_t> macroVecMap_t;
  typedef macroVecMap_t::iterator              macroVecMapIterator;
  typedef macroVecMap_t::const_iterator        cMacroVecMapIterator;

  typedef std::map<std::string, macro_t>       macroMap_t;
  typedef macroMap_t::iterator                 macroMapIterator;
  typedef macroMap_t::const_iterator           cMacroMapIterator;

  typedef void (*processDefine_t)(const char *&c);

  class preprocessor_t {
  public:
    enum status_t {
      reading       = (1 << 0),

      ignoring      = (3 << 1),
      ignoringIf    = (1 << 1),
      ignoringIfdef = (1 << 2)
    };

    hash_t compilerHash;
    std::string language;

    macroMap_t compilerMacros;
    macroMap_t languageMacros;
    macroVecMap_t macros;

    std::vector<stringChunk> content;
    const char *contentMark;

    std::stack<status_t> statusStack;

    trie_t<processDefine_t>& getDefineTrie() {
      static trie_t<processDefine_t> trie;
      if (trie.empty()) {
        trie.add("if"    , processIf);
        trie.add("elif"  , processElif);
        trie.add("else"  , processElse);
        trie.add("ifdef" , processIfdef);
        trie.add("ifndef", processIfndef);
        trie.add("endif" , processEndif);

        trie.add("define", processDefine);
        trie.add("undef" , processUndef);

        trie.add("error"  , processError);
        trie.add("warning", processWarning);

        trie.add("include", processInclude);
        trie.add("pragma" , processPragma);
        trie.add("line"   , processLine);
      }
      return trie;
    }

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
      const std::string directive(cStart, c - cStart);
      skipTo(c, '\n');
    }

    void processIf(const char *&c) {
      status_t status = getStatus();
      if (status & ignoring) {
        statusStack.push(ignoringIf);
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

    void processElse(const char *&c) {
      status_t status = getStatus();
      if (status & ignoring) {
        statusStack.push(ignoringAll);
        return;
      }
    }

    void processIfdef(const char *&c) {
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

    void processEndif(const char *&c) {
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

    void processUndef(const char *&c) {
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

    void processWarning(const char *&c) {
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

    void processPragma(const char *&c) {
      status_t status = getStatus();
      if (status & ignoring) {
        return;
      }
    }

    void processLine(const char *&c) {
      status_t status = getStatus();
      if (status & ignoring) {
        return;
      }
    }
  };
}
#  endif
#endif
