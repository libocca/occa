#ifndef OCCA_PARSER_PARSER_HEADER2
#define OCCA_PARSER_PARSER_HEADER2

#  if 0
#include <stack>

typedef std::vector<astNode*>           astNodeVector_t;
typedef astNodeVector_t::iterator       astNodeVectorIterator;
typedef astNodeVector_t::const_iterator cAstNodeVectorIterator;

typedef std::vector<lineNode>            lineNodeVector_t;
typedef lineNodeVector_t::iterator       lineNodeVectorIterator;
typedef lineNodeVector_t::const_iterator cLineNodeVectorIterator;

class lineNode {
  std::string value;
  int lineNumber;
  bool carriesOver;

  lineNode(const std::string &value_, const int lineNumber, const bool carriesOver_);
}

class astNode {
public:
  std::string value;
  astNode *up;
  astNodeVector_t leaves;
  int lineNumber;

  astNode(std::string value_, int lineNumber_);

  astNode& addLeaf(astNode *node);
  astNode& addLeaf(astNode &node);
};

class preprocessor_t {
  std::vector<macro_t> languageMacros;
  std::vector<macro_t> macros;

  preprocessStatus_t status = statusStack.top();

  void clear();
  void processLines(astNode &root);
};

preprocessor_t preprocessor;

astNode parse(const std::string &content) {
  preprocessor.clear();
  preprocessor.processLines(content);

  lineNodeVector_t lineNodes;
  splitLines(lineNodes, content.c_str());

}

void splitLines(lineNodeVector_t &lineNodes, const char *cRoot) {
  const char *c = cRoot;
  const char escapeChar = '\\';

  readingStatus_t status = readingCode;
  int lineNumber = 0;
  std::string line;

  while (*c != '\0') {
    const char *cStart = c;
    if (status == insideCommentBlock) {
      skipTo(c, "*/");
    } else {
      skipTo(c, '\n');
    }
    line += std::string(cStart, c - cStart);
    ++lineNumber;

    if (line.size() == 0) {
      continue;
    }

    // Line carrying over to next line
    const bool lineCarriesOver = ((line.size() > 1) &&
                                  (line[line.size() - 2] == continuationChar));
    if (lineCarriesOver) {
      line.erase(line.size() - 2);
    }

    status = stripComments(line);
    compressAllWhitespace(line);

    if (line.size()) {
      lineNodes.push_back(lineNode(line, lineNumber, lineCarriesOver));
      line.clear();
    }
    ++lineNumber;
  }
}

void preprocessor_t::clear() {
  macros.clear();
}

void preprocessor_t::processLines(astNode &root) {
  while (*c != '\0') {
    const char *cStart = c;
    skipTo(c, '\n');

    preprocessStatus_t status = statusStack.top();
    std::string &line = (*it)->value;
    bool ignoreLine = false;

    if (line[0] == '#') {
      preprocessStatus_t oldStatus = status;
      status = loadMacro(*it, status);
      ignoreLine = true;

      // Nested #if's
      if (status & startHash) {
        status &= ~startHash;
        statusStack.push(oldStatus);
      }

      if (status & doneIgnoring) {
        if (statusStack.size()) {
          statusStack.pop();
        } else {
          status = doNothing;
        }
      }
    } else {
      if (!(status & ignoring)) {
        applyMacros(line);
      } else {
        ignoreLine = true;
      }
    }

    if (ignoreLine) {
      ignoreLines.push_back(it);
    }

    statusStack.top() = status;
    ++it;
  }
}
#  endif
#endif