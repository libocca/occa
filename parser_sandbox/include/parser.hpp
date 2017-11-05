#ifndef OCCA_PARSER_PARSER_HEADER2
#define OCCA_PARSER_PARSER_HEADER2

// statement &root = parser.parseFile("filename");
// statement &root = parser.parse("const ...");
// statementVec_t kernels         = root["@kernel"];
// statementVec_t loops           = root["for"];
// statementVec_t innerLoops      = root["for@inner"];
// statementVec_t firstInnerLoops = root["for@inner:inner-most"];

#include <ostream>
#include <vector>
#include <map>

#include "occa/defines.hpp"
#include "occa/types.hpp"

#include "type.hpp"
#include "scope.hpp"
#include "statement.hpp"

namespace occa {
  class parser {
    scope_t globalScope, currentScope;

    type& getType(const std::string &name) {
      return currentScope.getType(name);
    }

    qualifier& getQualifier(const std::string &name) {
      return currentScope.getQualifier(name);
    }

    statement& parseFile(const std::string &filename) {
      char *content = io::c_read(filename);
      statement &root = parse(content);
      delete [] content;
      return root;
    }

    statement& parse(const std::string &content) {
      return parse(content.c_str());
    }

    statement& parse(const char *content) {
      tokenVec_t tokens;
      tokenize(content, tokens);

      blockStatement_t &root = *(new blockStatement_t());
      loadTokens(tokens, root);
      return root;
    }

    void tokenize(const char *content,
                  tokenVec_t &tokens) {
    }
}
#endif
