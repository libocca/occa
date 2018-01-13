/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
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
